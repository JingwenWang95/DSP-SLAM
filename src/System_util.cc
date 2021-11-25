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

#include "System.h"

namespace ORB_SLAM2
{

void System::SaveMapCurrentFrame(const string &dir, int frameId) {
    stringstream ss;
    ss << setfill('0') << setw(6) << frameId;

    string fname_pts = dir + "/" + ss.str() + "-MapPoints.txt";
    ofstream f_pts;
    f_pts.open(fname_pts.c_str());
    f_pts << fixed;
    const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
        if (vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        f_pts << setprecision(9) << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
    }
    f_pts.close();

    string fname_objects = dir + "/" + ss.str() + "-MapObjects.txt";
    ofstream f_obj;
    f_obj.open(fname_objects.c_str());
    f_obj << fixed;
    auto mvpMapObjects = mpMap->GetAllMapObjects();
    sort(mvpMapObjects.begin(), mvpMapObjects.end(), MapObject::lId);
    for (MapObject *pMO : mvpMapObjects) {
        if (!pMO)
            continue;
        if (pMO->isBad())
            continue;
        if (pMO->mRenderId < 0)
            continue;
        if (pMO->isDynamic())
            continue;

        f_obj << pMO->mnId << endl;
        auto Two = pMO->GetPoseSim3();
        f_obj << setprecision(9) << Two(0, 0) << " " << Two(0, 1) << " " << Two(0, 2) << " " << Two(0, 3) << " " <<
              Two(1, 0) << " " << Two(1, 1) << " " << Two(1, 2) << " " << Two(1, 3) << " " <<
              Two(2, 0) << " " << Two(2, 1) << " " << Two(2, 2) << " " << Two(2, 3) << endl;
        f_obj << setprecision(9) << pMO->GetShapeCode().transpose() << endl;
    }
    f_obj.close();

    vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
    string fname_keyframes = dir + "/" + ss.str() + "-KeyFrames.txt";
    ofstream f_kfs;
    f_kfs.open(fname_keyframes.c_str());
    f_kfs << fixed;
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];

        if (pKF->isBad())
            continue;

        cv::Mat Tcw = pKF->GetPose();
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        f_kfs << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2)
              << " " << twc.at<float>(0) << " " <<
              Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
              << twc.at<float>(1) << " " <<
              Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
              << twc.at<float>(2) << endl;

    }
    f_kfs.close();

    string fname_frame = dir + "/" + ss.str() + "-Camera.txt";
    ofstream f_camera;
    f_camera.open(fname_frame.c_str());
    f_camera << fixed;
    cv::Mat Tcw = mpTracker->mCurrentFrame.mTcw;
    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
    f_camera << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2)
             << " " << twc.at<float>(0) << " " <<
             Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " "
             << twc.at<float>(1) << " " <<
             Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " "
             << twc.at<float>(2) << endl;
    f_camera.close();

    cv::Mat frame = mpViewer->GetFrame();
    cv::imwrite(dir + "/" + ss.str() + "-Frame.png", frame);
}

void System::SaveEntireMap(const string &dir) {
    string fname_pts = dir + "/MapPoints.txt";
    ofstream f_pts;
    f_pts.open(fname_pts.c_str());
    f_pts << fixed;

    const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
        if (vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        f_pts << setprecision(9) << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
    }

    string fname_objects = dir + "/MapObjects.txt";
    ofstream f_obj;
    f_obj.open(fname_objects.c_str());
    f_obj << fixed;
    auto mvpMapObjects = mpMap->GetAllMapObjects();
    sort(mvpMapObjects.begin(), mvpMapObjects.end(), MapObject::lId);
    for (MapObject *pMO : mvpMapObjects) {
        if (!pMO)
            continue;
        if (pMO->isBad())
            continue;
        if (pMO->GetRenderId() < 0)
            continue;
        if (pMO->isDynamic())
            continue;

        f_obj << pMO->mnId << endl;
        auto Two = pMO->GetPoseSim3();
        f_obj << setprecision(9) << Two(0, 0) << " " << Two(0, 1) << " " << Two(0, 2) << " " << Two(0, 3) << " " <<
              Two(1, 0) << " " << Two(1, 1) << " " << Two(1, 2) << " " << Two(1, 3) << " " <<
              Two(2, 0) << " " << Two(2, 1) << " " << Two(2, 2) << " " << Two(2, 3) << endl;
        f_obj << setprecision(9) << pMO->GetShapeCode().transpose() << endl;
    }
    f_obj.close();

    SaveTrajectoryKITTI(dir + "/Cameras.txt");
}

}