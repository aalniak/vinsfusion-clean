#FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
#Image below is acquired by jetson-containers build with pytorch tag, with following command:
#jetson-containers build l4t-pytorch --skip-packages torch2trt
FROM l4t-pytorch:r36.4.tegra-aarch64-cu126-22.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    ROS_DISTRO=noetic \
    ROS_INSTALL_PREFIX=/opt/ros/noetic \
    VENV_PATH=/opt/venv \
    # Make venv take precedence and avoid user-site leakage (mismatch cause)
    PATH=/opt/venv/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


# Add ROS apt source for focal (needed for keys/rosdep metadata); import key
# (Using apt-key here for parity with your script inside a container context)
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Base tools and system deps (curl/wget/gnupg/equivs/build tools, python helpers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg lsb-release \
      git build-essential cmake pkg-config \
      python3-pip python3-venv python3-rosdep \
      python3-rosinstall-generator python3-vcstools python3-vcstool \
      python3-empy python3-catkin-pkg python3-rospkg python3-osrf-pycommon \
      python3-catkin-tools \
      libssl-dev libbz2-dev liblz4-tool \
      equivs && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_PATH}" && \
    "${VENV_PATH}/bin/python" -m ensurepip --upgrade && \
    "${VENV_PATH}/bin/python" -m pip install -U pip wheel setuptools && \
    # show where pip is to confirm it’s the venv one
    "${VENV_PATH}/bin/python" -m pip --version

# ---------------------------------------------
# Installation of ROS1 Noetic on Jammy (22.04) 
# ---------------------------------------------


# rosdep: initialize and patch sources (using the patch repo)
RUN set -euo pipefail; \
    rosdep init || true; \
    cd ~; \
    git clone https://github.com/aalniak/ROS1_patches.git; \
    mkdir -p ~/fake-hddtemp && cd ~/fake-hddtemp; \
    equivs-control hddtemp; \
    rm ./hddtemp; \
    cp ~/ROS1_patches/hddtemp ./hddtemp; \
    equivs-build hddtemp; \
    apt-get update && apt-get install -y ./hddtemp_0.3_all.deb; \
    mkdir -p ~/Downloads && cd ~/Downloads; \
    wget -q https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/base.yaml; \
    cp ~/ROS1_patches/base.yaml /tmp/base.yaml; \
    install -m 644 ~/ROS1_patches/20-default.list /etc/ros/rosdep/sources.list.d/20-default.list; \
    rosdep update

# Build Noetic from source with the jammy fixes (rosconsole / urdf branches)
RUN set -euo pipefail; \
    mkdir -p ~/ros_catkin_ws/src && cd ~/ros_catkin_ws; \
    rosinstall_generator desktop --rosdistro ${ROS_DISTRO} --deps --tar > noetic-desktop.rosinstall; \
    vcs import --input noetic-desktop.rosinstall ./src; \
    # Remove upstream rosconsole/urdf to replace with patched branches
    cd ~/ros_catkin_ws/src; \
    mkdir -p ~/Downloads/backup; \
    mv rosconsole urdf ~/Downloads/backup/; \
    git clone https://github.com/dreuter/rosconsole.git; \
    (cd rosconsole && git checkout noetic-jammy); \
    git clone https://github.com/dreuter/urdf.git; \
    (cd urdf && git checkout set-cxx-version); \
    # Install deps (ignore source pkgs)
    cd ~/ros_catkin_ws; \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} -y; \
    # Build and install Noetic
    /usr/bin/python3 ./src/catkin/bin/catkin_make_isolated \
      --install --install-space ${ROS_INSTALL_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=/usr/bin/python3

# Make ROS env available in shells
RUN echo "source ${ROS_INSTALL_PREFIX}/setup.bash" > /etc/profile.d/ros-noetic.sh

# -----------------------------
# Tolga tools
# -----------------------------

# Handy tools
RUN apt-get update && apt-get install -y --no-install-recommends tmux nano && \
    rm -rf /var/lib/apt/lists/*

# LLVM 18 repo + clang tools 
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg ca-certificates \
    && mkdir -p /usr/share/keyrings \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
         | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
         > /etc/apt/sources.list.d/llvm-18.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         clang-18 clang-format-18 clang-tidy-18 clangd-18 clang-tools-18 \
    && rm -rf /var/lib/apt/lists/*

# Set convenient alternatives (these binaries exist when installing the -18 packages)
RUN update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-18 100 && \
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-18   100 && \
    update-alternatives --install /usr/bin/run-clang-tidy run-clang-tidy /usr/bin/run-clang-tidy-18 100 && \
    update-alternatives --install /usr/bin/clang-apply-replacements clang-apply-replacements /usr/bin/clang-apply-replacements-18 100 && \
    update-alternatives --install /usr/bin/clangd       clangd       /usr/bin/clangd-18       100

# Ceres (CPU-only) — same as your original Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
      libatlas-base-dev libeigen3-dev libgoogle-glog-dev libsuitesparse-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone --depth 1 --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver && \
    cd /ceres-solver && mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCERES_ENABLE_CUDA=OFF \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF && \
    make -j"$(nproc)" && make install

# Prep an empty catkin workspace (you’ll mount your sources later)
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

RUN bash -lc '\
  source /etc/profile.d/ros-noetic.sh && \
  mkdir -p ~/catkin_ws/src && cd ~/catkin_ws && \
  catkin init && \
  catkin config --extend /opt/ros/noetic --merge-devel && \
  catkin config --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3 \
'

# Copy our Jetson-friendly requirements into the image
COPY docker/depth_requirements.txt /tmp/requirements.txt
# Install them into the venv (VENV_PATH=/opt/venv from above)
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install torch && \
    pip uninstall -y empy em && \
    pip install catkin_pkg "empy<4" rospkg

# Install tensorrt and torch2trt externally
RUN apt-get update && apt-get install -y gnupg wget && \
    wget -qO - https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/nvidia-l4t-apt-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t-apt-keyring.gpg] https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t.list && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t-apt-keyring.gpg] https://repo.download.nvidia.com/jetson/t234 r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t.list

RUN apt-get update && \
    apt-get install -y python3-libnvinfer && \
    # Search /usr/lib for the 'tensorrt' folder and copy it to venv
    find /usr/lib -type d -name "tensorrt" -exec cp -r {} /opt/venv/lib/python3.10/site-packages/ \; && \
    # Search for the egg-info metadata and copy it to venv (required for pip to see it)
    find /usr/lib -name "tensorrt-*.egg-info" -exec cp -r {} /opt/venv/lib/python3.10/site-packages/ \; || true && \
    rm -rf /var/lib/apt/lists/*
    
#RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
#    cd torch2trt && \
#    echo "/usr/lib/python3/dist-packages" > /opt/venv/lib/python3.10/site-packages/trt_link.pth && \
#    python3 -m pip install . --no-build-isolation && \
#    cd .. && \
#    rm -rf torch2trt

WORKDIR /root/catkin_ws
CMD ["/bin/bash"]
