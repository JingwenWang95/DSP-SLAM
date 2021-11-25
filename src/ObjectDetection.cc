/**
* This file is part of https://github.com/JingwenWang95/DSP-SLAM
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#include "ObjectDetection.h"

namespace ORB_SLAM2
{

// Detection from stereo+LiDAR measurement
ObjectDetection::ObjectDetection(const Eigen::Matrix4f &T, const Eigen::MatrixXf &Pts, const Eigen::MatrixXf &Rays,
                                 const Eigen::VectorXf &Depth)
{
    Sim3Tco = T;

    Rco = T.topLeftCorner<3, 3>();
    // scale is fixed once the object is initialized
    scale = pow(Rco.determinant(), 1. / 3.);
    Rco /= scale;
    tco = T.topRightCorner<3, 1>();
    // Transformation Matrix in SE3
    SE3Tco = Eigen::Matrix4f::Identity();
    SE3Tco.topLeftCorner<3, 3>() = Rco;
    SE3Tco.topRightCorner<3, 1>() = tco;

    SurfacePoints = Pts;
    RayDirections = Rays;
    DepthObs = Depth;
    nPts = (int) SurfacePoints.size() / 3;
    nRays = (int) RayDirections.size() / 3;
    isNew = true;
    isGood = true;
}

ObjectDetection::ObjectDetection() : isNew(true), isGood(true)
{
    Sim3Tco = Eigen::Matrix4f::Identity();
    Rco = Sim3Tco.topLeftCorner<3, 3>();
    scale = 1.;
    Rco /= scale;
    tco = Sim3Tco.topRightCorner<3, 1>();
    // Transformation Matrix in SE3
    SE3Tco = Eigen::Matrix4f::Identity();
    SE3Tco.topLeftCorner<3, 3>() = Rco;
    SE3Tco.topRightCorner<3, 1>() = tco;

    nPts = 0;
    nRays = 0;
}

std::vector<int> ObjectDetection::GetFeaturePoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvKeysIndices;
}

void ObjectDetection::AddFeaturePoint(const int &i)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvKeysIndices.push_back(i);
}

int ObjectDetection::NumberOfPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvKeysIndices.size();
}

void ObjectDetection::SetPoseMeasurementSim3(const Eigen::Matrix4f &T) {
    Rco = T.topLeftCorner<3, 3>();
    // scale is fixed once the object is initialized
    scale = pow(Rco.determinant(), 1. / 3.);
    Rco /= scale;
    tco = T.topRightCorner<3, 1>();
    // Transformation Matrix in SE3
    SE3Tco = Eigen::Matrix4f::Identity();
    SE3Tco.topLeftCorner<3, 3>() = Rco;
    SE3Tco.topRightCorner<3, 1>() = tco;
}

void ObjectDetection::SetPoseMeasurementSE3(const Eigen::Matrix4f &T) {
    SE3Tco = T;
    Rco = SE3Tco.topLeftCorner<3, 3>();
    tco = SE3Tco.topRightCorner<3, 1>();
    Sim3Tco.topLeftCorner<3, 3>() = scale * Rco;
    Sim3Tco.topRightCorner<3, 1>() = tco;
}
}