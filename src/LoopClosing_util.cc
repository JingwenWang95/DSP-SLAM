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

#include "LoopClosing.h"
#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include<mutex>
#include<thread>


namespace ORB_SLAM2 {

void LoopClosing::CorrectLoopWithObjects() {
    cout << "Loop detected with objects!" << endl;
    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA()) {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if (mpThreadGBA) {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped()) {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    // std::map<KeyFrame*, g2o::Sim3>
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end();
             vit != vend; vit++) {
            KeyFrame *pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if (pKFi != mpCurrentKF) {
                cv::Mat Tic = Tiw * Twc;
                cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                cv::Mat tic = Tic.rowRange(0, 3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi] = g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi] = g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end();
             mit != mend; mit++) {
            KeyFrame *pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            // Correct MapPoints
            vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++) {
                MapPoint *pMPi = vpMPsi[iMP];
                if (!pMPi)
                    continue;
                if (pMPi->isBad())
                    continue;
                if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();
            eigt *= (1. / s); //[R t/s;0 1]
            cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);
            auto NonCorrectedTiw = Converter::toMatrix4f(pKFi->GetPose());

            // Correct MapObjects
            vector<MapObject *> vpMOsi = pKFi->GetMapObjectMatches();
            for (MapObject *pMOi : vpMOsi) {
                if (!pMOi)
                    continue;
                if (pMOi->isBad())
                    continue;
                if (pMOi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                auto Tio = NonCorrectedTiw * pMOi->GetPoseSE3();
                auto CorrectedTwo = Converter::toMatrix4f(correctedTiw).inverse() * Tio;
                pMOi->SetObjectPoseSE3(CorrectedTwo);
                pMOi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMOi->mnCorrectedReference = pKFi->mnId;
            }

            // Update KeyFrame pose
            pKFi->SetPose(correctedTiw);
            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++) {
            if (mvpCurrentMatchedPoints[i]) {
                MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                if (pCurMP)
                    pCurMP->Replace(pLoopMP);
                else {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);
    SearchAndFuseObjects(CorrectedSim3);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame *, set<KeyFrame *> > LoopConnections;

    for (vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end();
         vit != vend; vit++) {
        KeyFrame *pKFi = *vit;
        vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        for (vector<KeyFrame *>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end();
             vit_prev != vend_prev; vit_prev++) {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (vector<KeyFrame *>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end();
             vit2 != vend2; vit2++) {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3,
                                      LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalJointBundleAdjustment, this, mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::SearchAndFuseObjects(const KeyFrameAndPose &CorrectedPosesMap) {
    // Retrieve MapObjects seen in Loop Keyframe and neighbors
    vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    vector<MapObject *> vpLoopObjects;
    for (KeyFrame *pKF : vpLoopConnectedKFs) {
        vector<MapObject *> vpMapObjects = pKF->GetMapObjectMatches();
        for (size_t i = 0, iend = vpMapObjects.size(); i < iend; i++) {
            MapObject *pMO = vpMapObjects[i];
            if (pMO) {
                if (!pMO->isBad() && pMO->mnLoopObjectForKF != mpCurrentKF->mnId) {
                    vpLoopObjects.push_back(pMO);
                    pMO->mnLoopObjectForKF = mpCurrentKF->mnId;
                }
            }
        }
    }

    // Project Loop KeyFrames MapObjects to the current KeyFrames and find matches
    for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end();
         mit != mend; mit++) {
        KeyFrame *pKF = mit->first;
        // Corrected KeyFrame pose
//        Eigen::Matrix4f Tcw = Converter::toMatrix4f(pKF->GetPose());
//        Eigen::Matrix3f Rcw = Tcw.topLeftCorner<3, 3>();
//        Eigen::Vector3f tcw = Tcw.topRightCorner<3, 1>();
        vector<MapObject *> vpReplaceObjects(vpLoopObjects.size(), static_cast<MapObject *>(NULL));
        const int nLO = vpLoopObjects.size();

        // Find objects in pKF that need to be replaced
        vector<MapObject *> vpLocalObjects = pKF->GetMapObjectMatches();
        for (int i = 0; i < nLO; i++) {
            MapObject *pLO = vpLoopObjects[i];
            vector<float> vDist;
            for (MapObject *pMOinKF : vpLocalObjects) {
                if (!pMOinKF) {
                    vDist.push_back(1e3);
                    continue;
                }
                if (pMOinKF->isBad()) {
                    vDist.push_back(1e3);
                    continue;
                }
                if (pMOinKF->isDynamic()) {
                    vDist.push_back(1e3);
                    continue;
                }

                // Compare under world frame should also work, as pMOinKF has been corrected already
                Eigen::Vector3f dist3D = pLO->two - pMOinKF->two;
                Eigen::Vector2f dist2D;
                dist2D << dist3D[0], dist3D[2];
                vDist.push_back((dist2D).norm());
            }

            float minDist = *min_element(vDist.begin(), vDist.end());
            if (minDist < 2.0) {
                int idx = min_element(vDist.begin(), vDist.end()) - vDist.begin();
                vpReplaceObjects[i] = vpLocalObjects[idx];
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        // Replace MapObjects
        for (int i = 0; i < nLO; i++) {
            MapObject *pRep = vpReplaceObjects[i];
            if (pRep) {
                pRep->Replace(vpLoopObjects[i]);
            }
        }
    }
}

void LoopClosing::RunGlobalJointBundleAdjustment(unsigned long nLoopKF) {
    cout << "Starting Global Joint Bundle Adjustment" << endl;

    int idx = mnFullBAIdx;
    Optimizer::GlobalJointBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

    // Update all MapPoints, KeyFrames and Objects
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx)
            return;

        if (!mbStopGBA) {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

            while (!lpKFtoCheck.empty()) {
                KeyFrame *pKF = lpKFtoCheck.front();
                const set<KeyFrame *> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for (set<KeyFrame *>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++) {
                    KeyFrame *pChild = *sit;
                    if (pChild->mnBAGlobalForKF != nLoopKF) {
                        cv::Mat Tchildc = pChild->GetPose() * Twc;
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF = nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

            for (size_t i = 0; i < vpMPs.size(); i++) {
                MapPoint *pMP = vpMPs[i];

                if (pMP->isBad())
                    continue;

                if (pMP->mnBAGlobalForKF == nLoopKF) {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                } else {
                    // Update according to the correction of its reference keyframe
                    KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                    cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                    cv::Mat twc = Twc.rowRange(0, 3).col(3);

                    pMP->SetWorldPos(Rwc * Xc + twc);
                }
            }

            // Correct MapPoints
            const vector<MapObject *> vpMOs = mpMap->GetAllMapObjects();

            for (size_t i = 0; i < vpMOs.size(); i++) {
                auto pMO = vpMOs[i];

                if (!pMO)
                    continue;
                if (pMO->isBad())
                    continue;
                if (pMO->isDynamic())
                    continue;

                if (pMO->mnBAGlobalForKF == nLoopKF) {
                    // If optimized by Global BA, just update
                    pMO->SetObjectPoseSE3(pMO->mTwoGBA);
                } else {
                    // Update according to the correction of its reference keyframe
                    KeyFrame *pRefKF = pMO->GetReferenceKeyFrame();
                    if (!pRefKF)
                        continue;
                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    auto TwoBefGBA = pMO->GetPoseSE3();
                    auto TcwBefGBA = Converter::toMatrix4f(pRefKF->mTcwBefGBA);
                    auto Tco = TcwBefGBA * TwoBefGBA;
                    auto Twc = Converter::toMatrix4f(pRefKF->GetPoseInverse());
                    auto Two = Twc * Tco;

                    pMO->SetObjectPoseSE3(Two);
                }
            }

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

}