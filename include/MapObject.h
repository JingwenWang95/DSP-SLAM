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

#ifndef MAPOBJECT_H
#define MAPOBJECT_H

#include <Eigen/Dense>
#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;

class MapObject {
public:
    MapObject(const Eigen::Matrix4f &T, const Eigen::Vector<float, 64> &vCode, KeyFrame *pRefKF, Map *pMap);
    MapObject(KeyFrame *pRefKF, Map *pMap);

    void AddObservation(KeyFrame *pKF, int idx);
    int Observations();
    std::map<KeyFrame*,size_t> GetObservations();
    void SetObjectPoseSim3(const Eigen::Matrix4f &Two);
    void SetObjectPoseSE3(const Eigen::Matrix4f &Two);
    void SetShapeCode(const Eigen::Vector<float, 64> &code);
    void UpdateReconstruction(const Eigen::Matrix4f &T, const Eigen::Vector<float, 64> &vCode);
    Eigen::Matrix4f GetPoseSim3();
    Eigen::Matrix4f GetPoseSE3();
    Eigen::Vector<float, 64> GetShapeCode();
    int GetIndexInKeyFrame(KeyFrame *pKF);
    void EraseObservation(KeyFrame *pKF);
    void SetBadFlag();
    bool isBad();
    void SetVelocity(const Eigen::Vector3f &v);
    void Replace(MapObject *pMO);
    bool IsInKeyFrame(KeyFrame *pKF);
    KeyFrame* GetReferenceKeyFrame();

    std::vector<MapPoint*> GetMapPointsOnObject();
    void AddMapPoints(MapPoint *pMP);
    void RemoveOutliersSimple();
    void RemoveOutliersModel();
    void ComputeCuboidPCA(bool updatePose);
    void EraseMapPoint(MapPoint *pMP);

    void SetRenderId(int id);
    int GetRenderId();
    void SetDynamicFlag();
    bool isDynamic();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix4f SE3Two;
    Eigen::Matrix4f SE3Tow;
    Eigen::Matrix4f Sim3Two;
    Eigen::Matrix4f Sim3Tow;
    Eigen::Matrix3f Rwo;
    Eigen::Vector3f two;
    float scale;
    float invScale;
    Eigen::Vector<float, 64> vShapeCode;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Reference KeyFrame
    KeyFrame *mpRefKF;
    KeyFrame *mpNewestKF;
    long unsigned int mnBALocalForKF;
    long unsigned int mnAssoRefID;
    long unsigned int mnFirstKFid;

    // variables used for loop closing
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    long unsigned int mnLoopObjectForKF;
    long unsigned int mnBAGlobalForKF;
    MapObject *mpReplaced;
    Eigen::Matrix4f mTwoGBA;

    bool reconstructed;
    std::set<MapPoint*> map_points;

    // cuboid
    float w;
    float h;
    float l;

    // Bad flag (we do not currently erase MapObject from memory)
    bool mbBad;
    bool mbDynamic;
    Eigen::Vector3f velocity;
    Map *mpMap;

    int nObs;
    static int nNextId;
    int mnId; // Object ID
    int mRenderId; // Object ID in the renderer
    Eigen::MatrixXf vertices;
    Eigen::MatrixXi faces;

    std::mutex mMutexObject;
    std::mutex mMutexFeatures;

    static bool lId(MapObject* pMO1, MapObject* pMO2){
        return pMO1->mnId < pMO2->mnId;
    }

};

}
#endif //MAPOBJECT_H
