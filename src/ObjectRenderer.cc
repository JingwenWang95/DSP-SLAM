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

#include "ObjectRenderer.h"

namespace ORB_SLAM2 {

Object::Object(const std::string &path_mesh) :
        mean(0, 0, 0),
        stddev(1, 1, 1),
        norm_factor(1) {
    pangolin::Geometry geometry = pangolin::LoadGeometry(path_mesh);
    object_gl = pangolin::ToGlGeometry(geometry);
}

Object::Object(const Eigen::MatrixXf &vertices, const Eigen::MatrixXi &faces) :
        mean(0, 0, 0),
        stddev(1, 1, 1),
        norm_factor(1) {
    if (vertices.cols() != 3) {
        throw std::invalid_argument("Only 3D vertices are supported");
    }
    if (faces.cols() != 3) {
        throw std::invalid_argument("Only triangular faces are supported");
    }

    size_t num_vertices = vertices.rows();
    size_t num_faces = faces.rows();
    pangolin::Geometry geometry;

    // Read vertices
    pangolin::Geometry::Element &geom_buffer = geometry.buffers["geometry"];
    geom_buffer.Reinitialise(3 * sizeof(float), num_vertices);
    pangolin::Image<float> verts = geom_buffer.UnsafeReinterpret<float>().SubImage(0, 0, 3, num_vertices);
    for (size_t i = 0; i < num_vertices; i++) {
        for (size_t j = 0; j < 3; j++) {
            verts(j, i) = vertices(i, j);
        }
    }
    geom_buffer.attributes["vertex"] = verts;

    // Read faces
    auto faces_buffer = geometry.objects.emplace("object", pangolin::Geometry::Element());
    faces_buffer->second.Reinitialise(3 * sizeof(uint32_t), num_faces);
    pangolin::Image<uint32_t> ibo = faces_buffer->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, num_faces);
    for (size_t f = 0; f < num_faces; ++f) {
        for (size_t v = 0; v < 3; ++v) {
            ibo(v, f) = faces(f, v);
        }
    }
    faces_buffer->second.attributes["vertex_indices"] = ibo;

    object_gl = pangolin::ToGlGeometry(geometry);
}

ObjectRenderer::ObjectRenderer(size_t w, size_t h, bool offscreen)
        : w(w), h(h) {
    if (offscreen)
        renderer = new Renderer(w, h);
    else
        renderer = new Renderer(0, 0);
}

void ObjectRenderer::SetupCamera(double fx, double fy, double cx, double cy, double near, double far) {
    this->fx = fx;
    this->fy = fy;
    this->cx = cx;
    this->cy = cy;
    this->near = near;
    this->far = far;
}

uint64_t ObjectRenderer::AddObject(const std::string &mesh_path) {
    uint64_t identifier = GetNextIdentifier();
    // objects[identifier] = std::make_shared<Object>(mesh_path);
    objects[identifier] = new Object(mesh_path);
    return identifier;
}

uint64_t ObjectRenderer::AddObject(const Eigen::MatrixXf &vertices, const Eigen::MatrixXi &faces) {
    uint64_t identifier = GetNextIdentifier();
    // objects[identifier] = std::make_shared<Object>(vertices, faces);
    objects[identifier] = new Object(vertices, faces);
    return identifier;
}

void
ObjectRenderer::Render(uint64_t identifier, const Eigen::Matrix4f &T_co, std::tuple<float, float, float> color) {
    auto o = objects.find(identifier);
    if (o == objects.end()) {
        throw std::invalid_argument("Unknown object, identifier: " + std::to_string(identifier));
    }
    Render(*o->second, identifier, T_co, color);
}

void ObjectRenderer::Render(const Object &object, uint64_t identifier, const Eigen::Matrix4f &T_co,
                            std::tuple<float, float, float> color) {

    renderer->Bind();
    renderer->SetUniformPinhole(w, h, fx, fy, cx, cy, T_co, near, far);
    renderer->SetUniformColor(std::get<0>(color), std::get<1>(color), std::get<2>(color));
    renderer->SetUniformMaskID(identifier);
    pangolin::GlDraw(renderer->program, object.object_gl, nullptr);
    renderer->Unbind();
}

}