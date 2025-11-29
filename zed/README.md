# ZED SDK compatibility
In order to work with ZED cameras, three main containers are required:

## 1. ZED ROS2 Container
ZED ROS2 wrapper shall be built from the [official zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper) repository, with the script under directory "docker" there by invoking the series of commands:
```bash
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
cd zed-ros2-wrapper/docker
./jetson_build_dockerfile_from_sdk_and_l4T_version.sh l4t-r36.4.0 zedsdk-5.1.0
```
Upon completion, your image repository:tag name is going to be used with the corresponding script. By default, it is dependent on script inputs, and is expected to be `zed_ros2_l4t_36.4.0_sdk_5.1.0:latest`. If there happens to be a difference, make sure to edit `zed.sh` to align it with your naming.

In `zed.sh`, the last mount argument tries to mount zed ros2 wrapper, to get any changed config from there (it's a design choice, you may change the config in the container as well under src).  
After considering these, and potentially double-checking `zed.sh` for your namings, you simply run:
```bash
bash zed.sh
```

## 2. ROS1 Master Container
This one is straightforward, as we only need a ROS1-capable container. We do not have any Ubuntu version dependence, so we use readily-available [ros:noetic-robot](https://hub.docker.com/layers/library/ros/noetic-robot/) image. One simply goes:
```bash
bash master.sh
```
After running, it detaches the roscore stuff for possible debug purposes with rostopic etc.  

## 3. ROS1_Bridge Container
As our system works in Ubuntu 22.04 (Jammy), and ZED SDK does not have ROS1 support for Jammy, [ros1_bridge](https://github.com/ros2/ros1_bridge) is an absolute necessity. It simply receives messages in ROS2 convention and transforms them into ROS1 then propagate to the ROS1 master.

The only nuance is that this container needs both ROS versions to be installed, and it is a little tedious to come up with one having both. We already had a pre-built Jammy Noetic container, which we again use it here. The dockerfile will build everything you need and will prepare it to go. Be careful on the available container naming again, as this tutorial follows the parent's naming convention (i.e. `noetic-on-jammy-l4t:latest`). First build the Dockerfile by:
```bash
docker build -t zed:ros_bridge -f Dockerfile.bridge .
```
This will take some time. After build finishes, one may run the bridge with:
```bash
bash bridge.sh
```
##
Once running all three, ROS1 master should be able to list zed topics with:
```bash
rostopic list
```
and the visible topics are directly usable in VINS-Fusion's configs.
