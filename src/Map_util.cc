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

#include "Map.h"
#include<mutex>

namespace ORB_SLAM2
{

void Map::AddMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.insert(pMO);
}

void Map::EraseMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.erase(pMO);
}

vector<MapObject*> Map::GetAllMapObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapObject*>(mspMapObjects.begin(), mspMapObjects.end());
}

MapObject* Map::GetMapObject(int object_id)
{
    unique_lock<mutex> lock(mMutexMap);
    for (auto mspMapObject : mspMapObjects)
    {
        if(mspMapObject->mnId != object_id)
            continue;
        return mspMapObject;
    }
    return NULL;
}

}

