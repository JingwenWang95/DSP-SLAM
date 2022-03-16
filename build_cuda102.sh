#!/bin/bash -e
#
# This is a build script for DSP-SLAM.
#
# Use parameters:
# `--install-cuda` to install the NVIDIA CUDA suite
#
# Example:
#   ./build_cuda102.sh --install-cuda --build-dependencies --create-conda-env
#
#   which will
#   1. Install some system dependencies
#   2. Install CUDA-11.3 under /usr/local
#   3. Create and build:
#   - ./Thirdparty/opencv
#   - ./Thirdparty/eigen
#   - ./Thirdparty/Pangolin
#   4. Build:
#   - ./Thirdparty/g2o
#   - ./Thirdparty/DBoW2
#   5. Create conda env with PyTorch 1.10
#   6. Install mmdetection and mmdetection3d
#   7. Build DSP-SLAM

# Function that executes the clone command given as $1 iff repo does not exist yet. Otherwise pulls.
# Only works if repository path ends with '.git'
# Example: git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
function git_clone(){
  repo_dir=`basename "$1" .git`
  git -C "$repo_dir" pull 2> /dev/null || eval "$1"
}

source Thirdparty/bashcolors/bash_colors.sh
function highlight(){
  clr_magentab clr_bold clr_white "$1"
}

highlight "Starting DSP-SLAM build script ..."
echo "Available parameters:
        --install-cuda
        --build-dependencies
        --create-conda-env"

highlight "Installing system-wise packages ..."
sudo apt-get update > /dev/null 2>&1 &&
sudo apt -y install gcc-8 g++-8 # gcc-8 is a safe version 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo apt-get install -y cmake
sudo apt-get install -y \
  libglew-dev \
  libgtk2.0-dev \
  pkg-config \
  libegl1-mesa-dev \
  libwayland-dev \
  libxkbcommon-dev \
  wayland-protocols

# install CUDA 10.2
if [[ $* == *--install-cuda* ]] ; then
  highlight "Installing CUDA..."
  wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
  sudo sh cuda_10.2.89_440.33.01_linux.run
  rm cuda_10.2.89_440.33.01_linux.run
fi # --install-cuda
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

if [[ $* == *--build-dependencies* ]]; then
  highlight "Installing OpenCV ..."
  cd Thirdparty
  git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
  cd opencv
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=OFF  \
      -DBUILD_DOCS=OFF  \
      -DBUILD_PACKAGE=OFF \
      -DBUILD_TESTS=OFF  \
      -DBUILD_PERF_TESTS=OFF  \
      -DBUILD_opencv_apps=OFF \
      -DBUILD_opencv_calib3d=ON  \
      -DBUILD_opencv_cudaoptflow=OFF  \
      -DBUILD_opencv_dnn=OFF  \
      -DBUILD_opencv_dnn_BUILD_TORCH_IMPORTER=OFF  \
      -DBUILD_opencv_features2d=ON \
      -DBUILD_opencv_flann=ON \
      -DBUILD_opencv_java=ON  \
      -DBUILD_opencv_objdetect=ON  \
      -DBUILD_opencv_python2=OFF  \
      -DBUILD_opencv_python3=OFF  \
      -DBUILD_opencv_photo=ON \
      -DBUILD_opencv_stitching=ON  \
      -DBUILD_opencv_superres=ON  \
      -DBUILD_opencv_shape=ON  \
      -DBUILD_opencv_videostab=OFF \
      -DBUILD_PROTOBUF=OFF \
      -DWITH_1394=OFF  \
      -DWITH_GSTREAMER=OFF  \
      -DWITH_GPHOTO2=OFF  \
      -DWITH_MATLAB=OFF  \
      -DWITH_NVCUVID=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_OPENCLAMDBLAS=OFF \
      -DWITH_OPENCLAMDFFT=OFF \
      -DWITH_TIFF=OFF  \
      -DWITH_VTK=OFF  \
      -DWITH_WEBP=OFF  \
      ..
  make -j8
  OpenCV_DIR=$(pwd)
  cd ../..

  highlight "Installing Eigen3 ..."
  git_clone "git clone --branch=3.4.0 --depth=1 https://gitlab.com/libeigen/eigen.git"
  cd eigen
  if [ ! -d build ]; then
    mkdir build
  fi
  if [ ! -d install ]; then
    mkdir install
  fi
  cd build
  cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/../install" ..
  make -j8
  make install
  cd ../..

  highlight "Installing Pangolin ..."
  git_clone "git clone --recursive --depth=1 https://github.com/stevenlovegrove/Pangolin.git"
  cd Pangolin
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  cmake ..
  make -j8
  Pangolin_DIR=$(pwd)
  cd ../..

  highlight "Installing g2o ..."
  cd g2o
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  cmake -DEigen3_DIR="$(pwd)/../../eigen/install/share/eigen3/cmake" ..
  make -j8
  cd ../..

  highlight "Installing DBoW2 ..."
  cd DBoW2
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  cmake -DOpenCV_DIR=$OpenCV_DIR ..
  make -j8
  cd ../../..
fi # --build-dependencies

if [[ $* == *--create-conda-env* ]] ; then
  highlight "Creating Python environment ..."
  conda env create -f environment_cuda102.yml
fi # --create-conda-env

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate dsp-slam

highlight "Installing mmdetection and mmdetection3d ..."
pip install pycocotools==2.0.1
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
cd Thirdparty
git_clone "git clone https://github.com/JingwenWang95/mmdetection3d.git"
cd mmdetection3d
pip install -v -e .
cd ../..

highlight "building DSP-SLAM ..."
if [ ! -d build ]; then
  mkdir build
fi
cd build
conda_python_bin=`which python`
conda_env_dir="$(dirname "$(dirname "$conda_python_bin")")"
cmake \
  -DOpenCV_DIR="$(pwd)/../Thirdparty/opencv/build" \
  -DEigen3_DIR="$(pwd)/../Thirdparty/eigen/install/share/eigen3/cmake" \
  -DPangolin_DIR="$(pwd)/../Thirdparty/Pangolin/build" \
  -DPYTHON_LIBRARIES="$conda_env_dir/lib/libpython3.7m.so" \
  -DPYTHON_INCLUDE_DIRS="$conda_env_dir/include/python3.7m" \
  -DPYTHON_EXECUTABLE="$conda_env_dir/bin/python3.7" \
  ..
make -j8

