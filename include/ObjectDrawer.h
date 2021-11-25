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

#ifndef OBJECTDRAWER_H
#define OBJECTDRAWER_H

#include "Map.h"
#include "MapObject.h"
#include "MapDrawer.h"
#include "KeyFrame.h"
#include "ObjectRenderer.h"
#include <pangolin/pangolin.h>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class MapDrawer;

class ObjectDrawer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ObjectDrawer(Map *pMap, MapDrawer *pMapDrawer, const string &strSettingPath);
    void SetRenderer(ObjectRenderer *pRenderer);
    void AddObject(MapObject *pMO);
    void ProcessNewObjects();
    void DrawObjects(bool bFollow, const Eigen::Matrix4f &Tec);
    void DrawCuboid(MapObject *pMO);
    void SetCurrentCameraPose(const Eigen::Matrix4f &Tcw);
    std::list<MapObject*> mlNewMapObjects;
    Map *mpMap;
    MapDrawer *mpMapDrawer;
    ObjectRenderer *mpRenderer;
    float mViewpointF;
    std::mutex mMutexObjects;
    std::vector<std::tuple<float, float, float>> mvObjectColors;
    Eigen::Matrix4f SE3Tcw; // current camera pose
    Eigen::Matrix4f SE3TcwFollow; // pose of camera which our eye is attached
};

}


#endif //OBJECTDRAWER_H
