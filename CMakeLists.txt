cmake_policy(SET CMP0048 NEW)
project("DSP-SLAM")
cmake_minimum_required(VERSION 3.2)

# c++ version
set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.2 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

# You need to set to your own Python paths
add_subdirectory(pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/System_util.cc
        src/Tracking.cc
        src/Tracking_util.cc
        src/LocalMapping.cc
        src/LocalMapping_util.cc
        src/LoopClosing.cc
        src/LoopClosing_util.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/MapPoint_util.cc
        src/KeyFrame.cc
        src/KeyFrame_util.cc
        src/Map.cc
        src/Map_util.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Optimizer_util.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/ObjectDetection.cc
        src/MapObject.cc
        src/ObjectDrawer.cc
        src/ObjectRenderer.cc
        )
target_link_libraries(${PROJECT_NAME}
        pybind11::embed
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam dsp_slam.cc)
target_link_libraries(dsp_slam ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam_mono dsp_slam_mono.cc)
target_link_libraries(dsp_slam_mono ${PROJECT_NAME})