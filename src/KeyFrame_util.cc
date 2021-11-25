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

#include "KeyFrame.h"

namespace ORB_SLAM2 {

void KeyFrame::AddMapObject(MapObject *pMO, int idx) {
    unique_lock<mutex> lock(mMutexObjects);
    mvpMapObjects[idx] = pMO;
}

void KeyFrame::EraseMapObjectMatch(const size_t &idx) {
    unique_lock<mutex> lock(mMutexObjects);
    mvpMapObjects[idx] = static_cast<MapObject *>(NULL);
}

void KeyFrame::EraseMapObjectMatch(MapObject *pMO) {
    int idx = pMO->GetIndexInKeyFrame(this);
    if (idx > 0)
        mvpMapObjects[idx] = static_cast<MapObject *>(NULL);
}

void KeyFrame::ReplaceMapObjectMatch(const size_t &idx, MapObject *pMO) {
    mvpMapObjects[idx] = pMO;
}

vector<ObjectDetection *> KeyFrame::GetObjectDetections() {
    unique_lock<mutex> lock(mMutexObjects);
    return mvpDetectedObjects;
}

vector<MapObject *> KeyFrame::GetMapObjectMatches() {
    unique_lock<mutex> lock(mMutexObjects);
    return mvpMapObjects;
}

}