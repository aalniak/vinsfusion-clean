#!/bin/bash

xhost +local:root
mkdir -p $HOME/.ws/vinsfusion
podman run \
    -it \
    --rm \
    --net=host \
    --privileged \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --device=/dev/dri \
    --device=/dev/video0 \
    -v "$HOME/.ws/vinsfusion/:/root/catkin_ws/" \
    -v "$(pwd)/:/root/catkin_ws/src/VINS-Fusion/" \
    -v "$1:/datasets/" \
    ros:vins-fusion \
    /bin/bash -c \
    "export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin:/opt/venv/bin; export CMAKE_PREFIX_PATH=/opt/ros/noetic; export OpenCV_DIR=/usr/local/lib/cmake/opencv4; cd /root/catkin_ws/; source /opt/ros/noetic/setup.bash; if [ -f devel/setup.bash ]; then source devel/setup.bash; fi; bash"
