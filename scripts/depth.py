#!/usr/bin/env python3
#
# ROS1 Depth Server for DepthAnything-AC
#
import os
import sys
import time
from typing import Tuple, Union

import cv2
import numpy as np
import torch

# --- ROS Imports ---
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# --- Project-Local Import (from your script) ---
# This logic assumes 'depth_server.py' is in '.../vins/scripts/'
# and the 'depth_anything' module is in '.../vins/depth_anything/'
# We add the package root ('.../vins/') to the Python path.
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

# --- Service Import ---
# (Replace 'vins' if your package name is different)
from vins.srv import EstimateDepth, EstimateDepthResponse


# ######################################################################
# ALL HELPER FUNCTIONS COPIED DIRECTLY FROM YOUR SCRIPT
# (No changes needed here)
# ######################################################################

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
    # Matches DepthAnything-AC config style
    return {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384],  "version": "v2"},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768], "version": "v2"},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "version": "v2"},
    }[encoder]


def preprocess_rgb(rgb: np.ndarray, target_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    rgb: HxWx3, uint8 RGB
    returns: (1,C,H’,W’) float32 normalized, original_size (H,W)
    """
    image = rgb.astype(np.float32) / 255.0
    h, w = image.shape[:2]

    # Scale so that min(h,w) -> target_size (preserve aspect)
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Make divisible by 14 (ViT patcher constraint)
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # HWC -> BCHW
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, (h, w)


def infer_disparity(model: torch.nn.Module, rgb: np.ndarray, input_size: int, device: str) -> np.ndarray:
    """
    Returns the RAW network output (disparity-like), resized back to original size (H,W), float32.
    """
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

    return disp.detach().float().cpu().numpy()  # (H,W) float32

# ######################################################################
# ROS SERVICE CLASS
# This replaces your main() and run_inference_single()
# ######################################################################

class DepthService:
    def __init__(self):
        rospy.loginfo("Starting DepthAnything ROS Service...")
        self.bridge = CvBridge()

        # --- 1. Get ROS Params (replaces argparse) ---
        # These params are "private" (~) and can be set in your .launch file
        default_encoder = "vits"
        self.encoder = rospy.get_param("~encoder", default_encoder)
        
        # Default model path. We make it relative to the PACKAGE_ROOT.
        if not rospy.has_param("~model_path"):
            rospy.logfatal("Missing ROS param '~model_path'.")
            rospy.logfatal("You MUST provide an absolute path to the model checkpoint.")
            rospy.logfatal('Example: <param name="model_path" value="/path/to/my/checkpoints/model.pth" />')
            return # Abort initialization
            
        self.model_path = rospy.get_param("~model_path")
        print("Model path is:", self.model_path)
        if not os.path.isabs(self.model_path):
            rospy.logwarn(f"Model path '{self.model_path}' is not absolute. This might fail.")
        
        self.input_size = rospy.get_param("~input_size", 210)
        cli_device = rospy.get_param("~device", None)  # e.g., "cuda:0" or "cpu"
        
        # --- 2. Load Model (from your run_inference_single) ---
        self.model = None
        self.device = pick_device(cli_device)
        rospy.loginfo(f"[Depth Server] Using device: {self.device}")

        try:
            cfg = get_model_config(self.encoder)
            self.model = DepthAnything_AC(cfg)
            
            rospy.loginfo(f"[Depth Server] Loading model from: {self.model_path}")
            if not os.path.exists(self.model_path):
                rospy.logfatal(f"[Depth Server] Model file NOT FOUND at {self.model_path}")
                rospy.logfatal("Please check the 'model_path' rosparam in your launch file.")
                return # Abort initialization

            ckpt = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(ckpt, strict=False)
            self.model = self.model.to(self.device).eval()
            rospy.loginfo("[Depth Server] Model loaded successfully.")

        except Exception as e:
            rospy.logfatal(f"[Depth Server] Failed to load model: {e}")
            return # Abort

        # --- 3. Advertise Service (only if model loaded) ---
        self.service = rospy.Service('estimate_depth', EstimateDepth, self.handle_request)
        rospy.loginfo("Depth estimation service is ready and waiting for requests.")

    def handle_request(self, req):
        """
        ROS Service callback.
        'req' is a vins.srv.EstimateDepthRequest object.
        It contains one field: 'input_image' (a sensor_msgs/Image)
        """
        start_time = rospy.get_time()
        
        # 1. Convert ROS Image msg -> OpenCV image (BGR)
        try:
            # Your script expects RGB, but cv_bridge default is BGR.
            # We'll get BGR and convert, just like your script does.
            cv_image_bgr = self.bridge.imgmsg_to_cv2(req.input_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (Input): {e}")
            return None # Return nothing to indicate failure

        # 2. Preprocess (BGR -> RGB)
        cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

        # 3. Run Inference
        try:
            # This is the core call to your function
            disparity_map = infer_disparity(
                self.model, 
                cv_image_rgb, 
                self.input_size, 
                self.device
            )
            # 'disparity_map' is a (H,W) float32 numpy array
        
        except Exception as e:
            rospy.logerr(f"Model inference failed: {e}")
            return None

        # 4. Convert OpenCV (numpy) -> ROS Image msg
        try:
            # We return the raw disparity map as a 32-bit float image (32FC1).
            # This is the most flexible format for your C++ VINS node.
            disparity_msg = self.bridge.cv2_to_imgmsg(disparity_map, encoding="32FC1")
            
            # Copy the header from the request to the response
            disparity_msg.header = req.input_image.header

            end_time = rospy.get_time()
            rospy.loginfo(f"Inference successful. Time: {end_time - start_time:.4f}s")

            # 5. Return the response object
            response = EstimateDepthResponse()
            response.depth_map = disparity_msg
            return response

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (Output): {e}")
            return None


if __name__ == "__main__":
    rospy.init_node('depth_server_node')
    try:
        ds = DepthService()
        # Only spin if the service was successfully created (i.e., model loaded)
        if hasattr(ds, 'service'):
            rospy.spin()
        else:
            rospy.logerr("DepthService failed to initialize. Shutting down node.")
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in depth server: {e}")
