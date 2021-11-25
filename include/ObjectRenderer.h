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

#ifndef OBJECTRENDERER_H
#define OBJECTRENDERER_H

#include "Renderer.hpp"
#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>

namespace ORB_SLAM2 {

class Object {
public:

    Object(const std::string &path_mesh);

    Object(const Eigen::MatrixXf &vertices, const Eigen::MatrixXi &faces);

    pangolin::GlGeometry object_gl;
    Eigen::Vector3f mean;
    Eigen::Vector3f stddev;
    float norm_factor;
};

// typedef std::shared_ptr<Object> ObjectPointer;

class ObjectRenderer {
public:
    ObjectRenderer(size_t w, size_t h, bool offscreen = false);

    void SetupCamera(double fx, double fy, double cx, double cy, double near, double far);

    uint64_t AddObject(const std::string &mesh_path);

    uint64_t AddObject(const Eigen::MatrixXf &vertices,
                       const Eigen::MatrixXi &faces);

    void Render(uint64_t identifier, const Eigen::Matrix4f &T_co, std::tuple<float, float, float> color);

    void Render(const Object &object, uint64_t identifier, const Eigen::Matrix4f &T_co,
                std::tuple<float, float, float> color);

    inline void DownloadColor(void *ptr_color) {
        renderer->DownloadColor(ptr_color);
    }

    inline int NumObjects() {
        return objects.size();
    }

    inline void Clear() {
        renderer->Clear();
    }

    inline size_t GetWidth() const {
        return renderer->GetWidth();
    }

    inline size_t GetHeight() const {
        return renderer->GetHeight();
    }


private:
    inline uint64_t GetNextIdentifier() {
        if (objects.empty())
            return 0;
        return objects.rbegin()->first + 1;
    }

private:
    double fx, fy, cx, cy, near, far;
    size_t w, h;
    std::map<uint64_t, Object *> objects;
    Renderer *renderer;
};

}

#endif //OBJECTRENDERER_H