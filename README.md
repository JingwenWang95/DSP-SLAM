# DSP-SLAM
### [Project Page](https://jingwenwang95.github.io/dsp-slam/) | [Video](https://youtu.be/of4ANH24LP4) | [Video (Bilibili)](https://www.bilibili.com/video/BV1yf4y1H7ib/) | [Paper](https://arxiv.org/abs/2108.09481)
This repository contains code for DSP-SLAM, an object-oriented SLAM system that builds a rich and accurate joint map of dense 3D models for foreground objects, and sparse landmark points to represent the background. DSP-SLAM takes as input the 3D point cloud reconstructed by a feature-based SLAM system and equips it with the ability to enhance its sparse map with dense reconstructions of detected objects. Objects are detected via semantic instance segmentation, and their shape and pose are estimated using category-specific deep shape embeddings as priors, via a novel second order optimization.  Our object-aware bundle adjustment builds a pose-graph to jointly optimize camera poses, object locations and feature points. DSP-SLAM can operate at 10 frames per second on 3 different input modalities: monocular, stereo, or stereo+LiDAR.

More information and the paper can be found at our [project page](https://jingwenwang95.github.io/dsp-slam/).

[![Figure of DSP-SLAM](figures/teaser.png "Click me to see a video.")](https://youtu.be/spNJlU982L4)

## Publication

[DSP-SLAM: Object Oriented SLAM with Deep Shape Priors](https://arxiv.org/abs/2108.09481), Jingwen Wang, [Martin Rünz](https://www.martinruenz.de/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/), 3DV '21

If you find our work useful, please consider citing our paper:
```
@inproceedings{wang2021dspslam,
  author={Jingwen Wang and Martin Rünz and Lourdes Agapito},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  title={DSP-SLAM: Object Oriented SLAM with Deep Shape Priors},
  pages={1362--1371}
  year={2021}
  organization={IEEE}
}
```

## Important Updates
**[16 Mar 2022]**: Due to some major changes in mmdetection3d, some components (such as detector weights) in this project no longer works with the latest mmdetection3d. I have updated the building scripts so that everything follows exactly the same setting as when this project was released.

# 1. Prerequisites
We have conducted most experiments and testings in Ubuntu 18.04 and 20.04, but it should also be possible to compile in other versions. You also need a powerful GPU to run DSP-SLAM, we have tested with RTX-2080 and RTX-3080. 

## TL;DR
We provide two building scripts which will install all the dependencies and build DSP-SLAM for you. Jump to [here](#building-script) for more details. If you want to have a more flexible installation then please read through this section carefully and refer to those two scripts as guidance. 

## C++17
We have used many new features in C++17, so please make sure your C++ compiler supports C++17. For g++ versions, we have tested with g++-7, g++-8 and g++-9.

## OpenCV
We use [OpenCV](https://github.com/opencv/opencv) for image related operations. Please make sure you have at least version 3.2. We have tested with OpenCV 3.4.1.

## Eigen3
We use [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix operations. Please make sure your Eigen3 version is at least 3.4.0. There is known compilation errors for lower versions.

## Pangolin
Pangolin is used for visualization the reconstruction result. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## DBoW2 and g2o (included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## pybind11 (included in project root directory)
As our shape reconstruction is implemented in Python, we need to enable communication between C++ and Python using pybin11. It is added as a submodule in this project, you just need to make sure you specify option `--recursive` when cloning the repository.

## Python Dependencies
Our prior-based object reconstruction is implemented in Python with PyTorch, which also requires MaskRCNN and PointPillars for 2D and 3D detection. 
* Python3 (tested with 3.7 and 3.8) and PyTorch (tested with 1.10) with CUDA (tested with 11.3 and 10.2)
* [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* Others: addict, plyfile, opencv-python, open3d

Compiling and installing mmdetection3d will require `nvcc`, so you need to make sure the CUDA version installed using `conda` matches the CUDA installed under your `usr/local/cuda-*`. `e.g.` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.10, you need to install the prebuilt PyTorch with CUDA 10.2.

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/). We have provided two example environment files which have CUDA 10.2/11.3 and PyTorch 1.10 for your reference. If you have CUDA 10.2 or CUDA 11.3 installed in your `/usr/local`, you can use it to set up your Python environment:

```
conda env create -f environment.yml
conda activate dsp-slam
``` 

Then you will still need to install mmdetection and mmdetection3d mannually. More details instruction can be found [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/getting_started.md).

# 2. Building DSP-SLAM

Clone the repository:
```
git clone --recursive https://github.com/JingwenWang95/DSP-SLAM.git
```

## Building script
For your convenience, we provide a building script `build_cuda102.sh` and `build_cuda113.sh` which show step-by-step how DSP-SLAM is built and which dependencies are required. Those scripts will install everything for you including CUDA (version is specified in the script name) and assume you have CUDA driver (support at least CUDA 10.2) and Anaconda installed on your computer. You can select whichever you want. `e.g.` If you your GPU is RTX-30 series which doesn't support CUDA 10 you can try with the one with CUDA 11.3. 

You can simply run:

```
./build_cuda***.sh --install-cuda --build-dependencies --create-conda-env
```

and it will set up all the dependencies and build DSP-SLAM for you. If you want to have a more flexible installation (use your own CUDA and Pytorch, build DSP-SLAM with your own version of OpenCV, Eigen3, etc), Those scripts can also provide important guidance for you.

## CMake options:
When building DSP-SLAM the following CMake options are mandatory: `PYTHON_LIBRARIES`, `PYTHON_INCLUDE_DIRS`, `PYTHON_EXECUTABLE`. Those must correspond to the same Python environment where your dependencies (PyTorch, mmdetection, mmdetection3d) are installed. Make sure these are correctly specified!

Once you have set up the dependencies, you can build DSP-SLAM: 

```
# (assume you are under DSP-SLAM project directory)
mkdir build
cd build
cmake -DPYTHON_LIBRARIES={YOUR_PYTHON_LIBRARY_PATH} \
      -DPYTHON_INCLUDE_DIRS={YOUR_PYTHON_INCLUDE_PATH} \
      -DPYTHON_EXECUTABLE={YOUR_PYTHON_EXECUTABLE_PATH} \
      ..
make -j8
```

After successfully building DSP-SLAM, you will have **libDSP-SLAM.so**  at *lib* folder and the executables **dsp_slam** and **dsp_slam_mono** under project root directory.

# 3. Running DSP-SLAM

## Dataset
You can download the example sequences and pre-trained network model weights (DeepSDF, MaskRCNN, PointPillars) from [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabjw4_ucl_ac_uk/Eh3nHv6D-LZHkuny4iNOexQBGdDVxloM_nwbEZdxeRfStw?e=sYO1Ot). It contains example sequences of [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), [Freiburg Cars](https://github.com/lmb-freiburg/unsup-car-dataset) and [Redwood Chairs](http://redwood-data.org/3dscan/dataset.html?c=chair) dataset.

## Run dsp_slam and dsp_slam_mono

After obtaining the 2 binary executables, you will need to suppy 4 parameters to run the program: 1. path to vocabulary 2. path to .yaml config file 3. path to sequence data directory 4. path to save map. Before running DSP-SLAM, make sure you run `conda activate dsp-slam` to activate the correct Python environmrnt. Here are some example usages:

For KITTI sequence for example, you can run:

```
./dsp_slam Vocabulary/ORBvoc.bin configs/KITTI04-12.yaml data/kitti/07 map/kitti/07
```

For Freiburg Cars:

```
./dsp_slam_mono Vocabulary/ORBvoc.bin configs/freiburg_001.yaml data/freiburg/001 map/freiburg/001
```

For Redwood Chairs:

```
./dsp_slam_mono Vocabulary/ORBvoc.bin configs/redwood_09374.yaml data/redwood/09374 map/redwood/09374
```

## Save and visualize map

If you supply a valid path to DSP-SLAM as the 4-th argument, after running the program you should get 3 text files under that directory: Cameras.txt, MapObjects.txt and MapPoints.txt. MapObjects.txt stores the reconstructed object(s) as shape code and 7-DoF pose. Before you can visualize the map, you need to extract meshes from shape codes by running:

```
python extract_map_objects.py --config configs/config_kitti.json --map_dir map/07 --voxels_dim 64
```

It will create a new directory under map/07 and stores all the meshes and object poses there. Then you will be able to visualize the reconstructed joint map by running:

```
python visualize_map.py --config configs/config_kitti.json --map_dir map/07
```

Then you will be able to view the map in an Open3D window:

<p align="justify">
  <img src="figures/map-1.png" width="400"> <img src="figures/map-2.png" width="400">
</p>

## Tips
### Try python script of single-shot reconstruction first
We provide a Python script [reconstruct_frame.py](reconstruct_frame.py) which does 3D object reconstruction from a single frame for KITTI sequences. Running it does not require any C++ stuff. Here is an example usage:
```
python reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/kitti/07 --frame_id 100
```
If you can run it smoothly you will see a Open3D window pop up. The figure below shows an example result:

<p align="justify">
  <img src="figures/single-shot_recon.png" width="1000">
</p>

### Run DSP-SLAM with offline detector
If you can successfully build DSP-SLAM but get errors from Python side when running the program, then you can try supplying pre-stored labels and run DSP-SLAM with offline detector. We have provided 2D and 3D labels for KITTI sequence in the [data](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabjw4_ucl_ac_uk/Eh3nHv6D-LZHkuny4iNOexQBGdDVxloM_nwbEZdxeRfStw?e=sYO1Ot). To run DSP-SLAM with offline mode, you will need to change the field `detect_online` in the .json config file to `false` and specify the corresponding label path.

### Label format
If you want to create your own labels with your own detectors, you can follow the same format as the labels we provided in the KITTI-07 sequence.
* 3D labels contains 3D detection boxes under [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) convention. Each `.lbl` file consits of a numpy array of size Nx7, where N is the number of objects detected. Each row of the array is a 3D detection box: [x, y, z, w, l, h, ry]. More information about the KITTI coordinate system can be found from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) or [KITTI website](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf).
* 2D labels contains MaskRCNN detection boxes and segmentation masks. Each `.lbl` file consists of of a dictionary with two keys: `pred_boxes` and `pred_masks`. Boxes and masks are stored as numpy array of size Nx4 and NxHxW.

### Run DSP-SLAM with mono sequence
If you have problem installing mmdetection3d but can run mmdetection smoothly, then you can start with mono sequences as they only require 2D detector.

# 4. License
DSP-SLAM includes the third-party open-source software ORB-SLAM2, which itself includes third-party open-source software. Each of these components have their own license. DSP-SLAM also takes a part of code from DeepSDF which is under MIT license: https://github.com/facebookresearch/DeepSDF.

DSP-SLAM is released under [GPLv3 license](LICENSE) in line with ORB-SLAM2. For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](Dependencies.md).

# 5. FAQ

### Resolve '***Segmentation Fault***'
First make sure you have activated the correct environment which is `dsp-slam` and run DSP-SLAM under the project root directory. See if that can resolve the SegFault.

Another major cause of SegFault is caused by mmdetection3d. If you encounter SegFault on KITTI sequence but can successfully run `dsp_slam_mono` on Freiburg Cars or Redwood Chairs, then the problem is probably because mmdetection3d is not correctly installed. Make sure you fully understand the steps in the building scripts and followed all the steps correctly.

### No response after running
Make sure you specified correct data path when running DSP-SLAM. 

### Resolve '***The experiment directory does not include specifications file "specs.json"***'
This error suggests that you provided wrong path for DeepSDF weights. All the [json config files](https://github.com/JingwenWang95/DSP-SLAM/blob/master/configs/config_kitti.json#L19) under `configs/` assume that weights are under stored `weights/`. Make sure you download the weights and put them under `weights/` or change the weight path in the json config files.

### Resolve '***ValueError: numpy.ndarray size changed, may indicate binary incompatibility***'
This error is caused by numpy version mismatch between mmdetection3d and pycocotools. This can be resolved by upgrade numpy version to >= 1.20.0. More info can befound from here: https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp

### Resolve '***GLSL Shader compilation error: GLSL 3.30 is not supported***'
We render the reconstructed objects using GLSL, the error above won't affect the SLAM system running but you won't see objects show up in the window. You can try running ```export MESA_GL_VERSION_OVERRIDE=3.3``` before running DSP-SLAM. More info: https://stackoverflow.com/questions/52592309/0110-error-glsl-3-30-is-not-supported-ubuntu-18-04-c

# 6. Acknowledgements
Research presented here has been supported by the UCL Centre for Doctoral Training in Foundational AI under UKRI grant number EP/S021566/1. We thank [Wonbong Jang](https://sites.google.com/view/wbjang/home) and Adam Sherwood for fruitful discussions. We are also grateful to [Binbin Xu](https://www.doc.ic.ac.uk/~bx516/) and [Xin Kong](https://kxhit.github.io/) for their patient code testing!
