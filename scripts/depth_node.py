#!/usr/bin/env python3
#
# ROS1 Depth Node for DepthAnything-AC (asynchronous publisher)
#

import os
import sys
from typing import Tuple, Union
import threading
from queue import Queue, Full

import cv2
import numpy as np
import torch

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# --- Project-Local Import Setup (same as your service) ---
ABSOLUTE_MODULE_PATH = "/depth"
os.chdir("/depth")
try:
    sys.path.append(ABSOLUTE_MODULE_PATH)
    from depth_anything.dpt import DepthAnything_AC
    rospy.loginfo(f"Successfully imported DepthAnything_AC from: {ABSOLUTE_MODULE_PATH}")
except ImportError as e:
    print(e)
    rospy.logfatal(f"Failed to import DepthAnything_AC. Check sys.path.")
    rospy.logfatal(f"Attempted to add {ABSOLUTE_MODULE_PATH} to path.")
    rospy.logfatal("Is the 'depth_anything' folder located inside '/depth'?")
    sys.exit(1)


# ================== YOUR EXISTING HELPERS =====================

def pick_device(cli_device: Union[str, None]) -> str:
    if cli_device:
        return cli_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    print("We going cpu")
    return "cpu"


def get_model_config(encoder: str) -> dict:
    return {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384],  "version": "v2"},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768], "version": "v2"},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "version": "v2"},
    }[encoder]


def preprocess_rgb(rgb: np.ndarray, target_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]

    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, (h, w)


def infer_disparity(model: torch.nn.Module, rgb: np.ndarray, input_size: int, device: str) -> np.ndarray:
    inp, (h, w) = preprocess_rgb(rgb, input_size)
    inp = inp.to(device)
    with torch.no_grad():
        pred = model(inp)
        disp = pred["out"]
        if disp.dim() == 4 and disp.size(1) == 1:
            disp = torch.nn.functional.interpolate(disp, size=(h, w), mode="bilinear", align_corners=True)
            disp = disp.squeeze(0).squeeze(0)
        elif disp.dim() == 3:
            disp = torch.nn.functional.interpolate(disp.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True)
            disp = disp.squeeze(0).squeeze(0)
        else:
            disp = disp.view(1, 1, *disp.shape[-2:])
            disp = torch.nn.functional.interpolate(disp, size=(h, w), mode="bilinear", align_corners=True)
            disp = disp.squeeze(0).squeeze(0)

    return disp.detach().float().cpu().numpy()


# ================== NEW: ASYNC NODE =====================

class DepthNode:
    def __init__(self):
        self.bridge = CvBridge()

        # -------- Parameters --------
        # Same as your service:
        self.encoder = rospy.get_param("~encoder", "vits")

        if not rospy.has_param("~model_path"):
            rospy.logfatal("Missing ROS param '~model_path'. Please provide checkpoint path.")
            raise RuntimeError("model_path not set")

        self.model_path = rospy.get_param("~model_path")
        if not os.path.isabs(self.model_path):
            rospy.logwarn(f"Model path '{self.model_path}' is not absolute. This might fail.")

        self.input_size = rospy.get_param("~input_size", 210)
        cli_device      = rospy.get_param("~device", None)

        # New: input/output topics
        self.input_topic  = rospy.get_param("~input_topic", "/zed/left/image_rect_color")
        self.output_topic = rospy.get_param("~output_topic", "/mde/depth")

        # Queue & threading settings
        # We keep only the latest frame: maxsize=1 → prevents backlog and keeps video smooth.
        self.image_queue: Queue = Queue(maxsize=1)

        # -------- Load Model --------
        self.device = pick_device(cli_device)
        rospy.loginfo(f"[Depth Node] Using device: {self.device}")

        cfg = get_model_config(self.encoder)
        self.model = DepthAnything_AC(cfg)

        rospy.loginfo(f"[Depth Node] Loading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            rospy.logfatal(f"[Depth Node] Model file NOT FOUND at {self.model_path}")
            raise RuntimeError("Model file missing")

        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.to(self.device).eval()
        rospy.loginfo("[Depth Node] Model loaded successfully.")

        # -------- ROS I/O --------
        self.depth_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # -------- Worker Thread --------
        self.worker_thread = threading.Thread(target=self.depth_worker, daemon=True)
        self.worker_thread.start()

        rospy.loginfo(
            f"[Depth Node] Ready. Subscribing to '{self.input_topic}', "
            f"publishing depth to '{self.output_topic}'."
        )

    def image_callback(self, msg: Image):
        """Lightweight callback: just store the latest frame in the queue."""
        try:
            # If queue is full, drop the old one and insert the new frame.
            # This avoids lag buildup.
            if self.image_queue.full():
                try:
                    _ = self.image_queue.get_nowait()
                except Exception:
                    pass
            self.image_queue.put_nowait(msg)
        except Full:
            # In theory we shouldn't hit this after popping above, but just in case:
            pass

    def depth_worker(self):
        """Runs in a background thread, performs heavy inference."""
        rospy.loginfo("[Depth Node] Worker thread started.")

        while not rospy.is_shutdown():
            try:
                # Blocks until a frame is available
                msg: Image = self.image_queue.get()
            except Exception as e:
                rospy.logerr(f"[Depth Node] Failed to get image from queue: {e}")
                continue

            start_time = rospy.get_time()

            # Convert ROS -> OpenCV
            try:
                cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"[Depth Node] CvBridge Error (Input): {e}")
                self.image_queue.task_done()
                continue

            # BGR -> RGB for your model
            cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

            # Run model
            try:
                disparity_map = infer_disparity(
                    self.model,
                    cv_rgb,
                    self.input_size,
                    self.device,
                )
            except Exception as e:
                rospy.logerr(f"[Depth Node] Model inference failed: {e}")
                self.image_queue.task_done()
                continue

            # Convert back to ROS Image
            try:
                depth_msg = self.bridge.cv2_to_imgmsg(disparity_map, encoding="32FC1")
                depth_msg.header = msg.header  # preserve timestamp & frame_id

                self.depth_pub.publish(depth_msg)
                end_time = rospy.get_time()
                rospy.loginfo_throttle(
                    1.0,  # log at most once per second
                    f"[Depth Node] Published depth. Inference time: {end_time - start_time:.3f}s",
                )
            except CvBridgeError as e:
                rospy.logerr(f"[Depth Node] CvBridge Error (Output): {e}")

            self.image_queue.task_done()


if __name__ == "__main__":
    rospy.init_node("depth_anything_node")
    try:
        node = DepthNode()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"[Depth Node] Fatal error: {e}")
