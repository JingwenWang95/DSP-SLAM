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

#ifndef OBJECTPOSEGRAPH_H
#define OBJECTPOSEGRAPH_H

#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include <Eigen/Geometry>
#include "Converter.h"

namespace ORB_SLAM2 {

using namespace g2o;
using namespace std;

class VertexSE3Object : public BaseVertex<6, SE3Quat> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSE3Object() {};

    virtual bool read(std::istream &is) {
        return true;
    }

    virtual bool write(std::ostream &os) const {
        return os.good();
    }

    virtual void setToOriginImpl() {
        _estimate = SE3Quat();
    }

    // gradient is wrt Twc, but our estimate is Tcw
    virtual void oplusImpl(const double *update_) {
        Eigen::Map<Vector6d> update(const_cast<double *>(update_));
        SE3Quat s(update);
        setEstimate(estimate() * s.inverse());
    }
};

class EdgeSE3LieAlgebra : public BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(std::istream &is) {
        return true;
    }

    bool write(std::ostream &os) const {
        return os.good();
    }

    virtual void computeError() {
        SE3Quat v1 = (static_cast<VertexSE3Expmap *>(_vertices[0]))->estimate(); // Ti
        SE3Quat v2 = (static_cast<VertexSE3Expmap *>(_vertices[1]))->estimate(); // Tj
        _error = (_measurement.inverse() * v1 * v2.inverse()).log(); // Tij^-1 * Ti * Tj^-1
    }

    virtual void linearizeOplus() {
        Matrix6d J;
        Eigen::Vector3d t, w;
        w = _error.head<3>();
        t = _error.tail<3>();
        J.block<3, 3>(0, 0) = skew(w);
        J.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        J.block<3, 3>(3, 0) = skew(t);
        J.block<3, 3>(3, 3) = skew(w);
        J = 0.5 * J + Matrix6d::Identity();

        _jacobianOplusXi = J * _measurement.inverse().adj();
        _jacobianOplusXj = -J;
    }
};

}
#endif //OBJECTPOSEGRAPH_H
