/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Converter.h"

namespace ORB_SLAM2
{

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

g2o::SE3Quat Converter::toSE3Quat(const Eigen::Matrix4f &T)
{
    Eigen::Matrix<double, 3, 3> R = T.topLeftCorner<3, 3>().cast<double>();
    Eigen::Matrix<double, 3, 1> t = T.topRightCorner<3, 1>().cast<double>();
    return g2o::SE3Quat(R, t);
}

g2o::Sim3 Converter::toSim3(const Eigen::Matrix4f &T)
{
    Eigen::Matrix3d sR = T.topLeftCorner<3, 3>().cast<double>();
    double s = pow(sR.determinant(), 1. / 3);
    Eigen::Matrix3d R = sR / s;
    Eigen::Vector3d t = T.topRightCorner<3, 1>().cast<double>();
    return g2o::Sim3(R, t, s);
}

cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<float,3,1> Converter::toVector3f(const cv::Mat &cvVector)
{
    Eigen::Matrix<float,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix<float, 3, 3> Converter::toMatrix3f(const cv::Mat &cvMat3)
{
    Eigen::Matrix<float, 3, 3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
            cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
            cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix4f Converter::toMatrix4f(const cv::Mat &cvMat)
{
    Eigen::Matrix4f M;
    M << cvMat.at<float>(0,0), cvMat.at<float>(0,1), cvMat.at<float>(0,2), cvMat.at<float>(0, 3),
         cvMat.at<float>(1,0), cvMat.at<float>(1,1), cvMat.at<float>(1,2), cvMat.at<float>(1, 3),
         cvMat.at<float>(2,0), cvMat.at<float>(2,1), cvMat.at<float>(2,2), cvMat.at<float>(2, 3),
         cvMat.at<float>(3,0), cvMat.at<float>(3,1), cvMat.at<float>(3,2), cvMat.at<float>(3, 3);

    return M;
}

Eigen::Matrix4f Converter::toMatrix4f(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix4f eigMat = SE3.to_homogeneous_matrix().cast<float>();
    return eigMat;
}


Eigen::Matrix4f Converter::toMatrix4f(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f eigR = Sim3.rotation().toRotationMatrix().cast<float>();
    Eigen::Vector3f eigt = Sim3.translation().cast<float>();
    float s = Sim3.scale();
    T.topLeftCorner<3, 3>() = s * eigR;
    T.topRightCorner<3, 1>() = eigt;
    return T;
}

Eigen::Matrix4d Converter::toMatrix4d(const cv::Mat &cvMat)
{
    Eigen::Matrix4d M;
    M << cvMat.at<float>(0,0), cvMat.at<float>(0,1), cvMat.at<float>(0,2), cvMat.at<float>(0, 3),
            cvMat.at<float>(1,0), cvMat.at<float>(1,1), cvMat.at<float>(1,2), cvMat.at<float>(1, 3),
            cvMat.at<float>(2,0), cvMat.at<float>(2,1), cvMat.at<float>(2,2), cvMat.at<float>(2, 3),
            cvMat.at<float>(3,0), cvMat.at<float>(3,1), cvMat.at<float>(3,2), cvMat.at<float>(3, 3);

    return M;
}

pangolin::OpenGlMatrix Converter::toMatrixPango(const Eigen::Matrix4f &T)
{
    pangolin::OpenGlMatrix M;
    M.SetIdentity();

    M.m[0] = T(0,0);
    M.m[1] = T(1,0);
    M.m[2] = T(2,0);
    M.m[3]  = 0.0;

    M.m[4] = T(0,1);
    M.m[5] = T(1,1);
    M.m[6] = T(2,1);
    M.m[7]  = 0.0;

    M.m[8] = T(0,2);
    M.m[9] = T(1,2);
    M.m[10] = T(2,2);
    M.m[11]  = 0.0;

    M.m[12] = T(0,3);
    M.m[13] = T(1, 3);
    M.m[14] = T(2, 3);
    M.m[15]  = 1.0;

    return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

} //namespace ORB_SLAM
