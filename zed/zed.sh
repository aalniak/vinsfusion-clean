#!/usr/bin/env bash
set -euo pipefail
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

# Common ROS 2 env
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

docker run --runtime=nvidia --rm \
  --name zed2 \
  -it --privileged \
  --network host \
  -e DISPLAY -e XAUTHORITY \
  -e ROS_DOMAIN_ID=42 \
  -e RMW_IMPLEMENTATION -e ROS_LOCALHOST_ONLY \
  -v "$XAUTHORITY:$XAUTHORITY:ro" \
  -v /usr/local/zed/resources:/usr/local/zed/resources \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  -v /media/SSD/zed/zed-ros2-wrapper/zed_wrapper/config/:"/root/ros2_ws/install/zed_wrapper/share/zed_wrapper/config/" \
  zed_ros2_l4t_36.4.0_sdk_5.1.0 \
  bash -lc "source /opt/ros/humble/install/setup.bash && \
            source /root/ros2_ws/install/setup.bash && \
            export ROS_DOMAIN_ID=42 && \
            export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
            export ROS_LOCALHOST_ONLY=0 && \
            ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i"
