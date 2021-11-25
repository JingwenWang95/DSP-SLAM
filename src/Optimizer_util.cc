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

# include "Optimizer.h"
# include "Thirdparty/g2o/g2o/core/block_solver.h"
# include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
# include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
# include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
# include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
# include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
# include "Thirdparty/g2o/g2o/core/factory.h"
# include "ObjectPoseGraph.h"

namespace ORB_SLAM2
{

G2O_REGISTER_TYPE(VERTEX_SE3:OBJ, VertexSE3Object);
G2O_REGISTER_TYPE(EDGE_SE3:LIE_ALGEBRA, EdgeSE3LieAlgebra);

int Optimizer::nBAdone = 0;

void Optimizer::GlobalJointBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF,
                                       const bool bRobust) {
    vector < KeyFrame * > vpKFs = pMap->GetAllKeyFrames();
    vector < MapPoint * > vpMP = pMap->GetAllMapPoints();
    vector < MapObject * > vpMO = pMap->GetAllMapObjects();
    JointBundleAdjustment(vpKFs, vpMP, vpMO, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::JointBundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                      const std::vector<MapObject *> &vpMO,
                                      int nIterations, bool *pbStopFlag, const unsigned long nLoopKF,
                                      const bool bRobust) {
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());
    vector<bool> vbNotIncludedMO;
    vbNotIncludedMO.resize(vpMO.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    long unsigned int maxMPid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);
    const float invSigmaObject = 1e3;
    const float thHuberObject = sqrt(0.10 * invSigmaObject);
    const float thHuberObjectSquare = pow(thHuberObject, 2);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        if (pMP->mnId > maxMPid)
            maxMPid = pMP->mnId;

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            } else {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    // Set MapObject Vertices
    for (size_t i = 0; i < vpMO.size(); i++) {
        auto pMO = vpMO[i];

        if (!pMO) {
            vbNotIncludedMO[i] = true;
            continue;
        }
        if (pMO->isDynamic() || pMO->isBad()) {
            vbNotIncludedMO[i] = true;
            continue;
        }


        g2o::VertexSE3Expmap *vSE3Obj = new g2o::VertexSE3Expmap();
        vSE3Obj->setEstimate(Converter::toSE3Quat(pMO->SE3Tow));
        int id = pMO->mnId + maxKFid + maxMPid + 2;
        vSE3Obj->setId(id);
        optimizer.addVertex(vSE3Obj);

        const map<KeyFrame *, size_t> observations = pMO->GetObservations();

        // Set Edges
        int nEdges = 0;
        for (auto observation : observations) {
            KeyFrame *pKFi = observation.first;

            // reject those frames after requesting stop
            if (pKFi->isBad() || pKFi->mnId > maxKFid)
                continue;
            // Get detections
            auto mvpObjectDetections = pKFi->GetObjectDetections();

            // cout << "Object KF ID: " << pKFi->mnId << endl;
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            e->setVertex(0, optimizer.vertex(pKFi->mnId));
            e->setVertex(1, optimizer.vertex(id));
            auto det = mvpObjectDetections[observation.second];
            e->setMeasurement(Converter::toSE3Quat(det->SE3Tco));
            Eigen::Matrix<double, 6, 6> Info = Eigen::Matrix<double, 6, 6>::Identity();
            Info *= invSigmaObject;
            e->setInformation(Info);

            if (bRobust) {
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberObject);
            }

            optimizer.addEdge(e);
            nEdges++;
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vSE3Obj);
            vbNotIncludedMO[i] = true;
        } else {
            vbNotIncludedMO[i] = false;
        }

    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                pMP->mnId + maxKFid + 1));

        if (nLoopKF == 0) {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        } else {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

    // Objects
    for (size_t i = 0; i < vpMO.size(); i++) {
        if (vbNotIncludedMO[i])
            continue;

        MapObject *pMO = vpMO[i];

        if (pMO->isBad())
            continue;
        if (pMO->isDynamic())
            continue;

        g2o::VertexSE3Expmap *vSE3Obj = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(
                pMO->mnId + maxKFid + maxMPid + 2));
        g2o::SE3Quat SE3Tow = vSE3Obj->estimate();

        if (nLoopKF == 0) {
            Eigen::Matrix4f SE3Two = Converter::toMatrix4f(SE3Tow).inverse();
            pMO->SetObjectPoseSE3(SE3Two);
        } else {
            pMO->mTwoGBA = Converter::toMatrix4f(SE3Tow).inverse();
            pMO->mnBAGlobalForKF = nLoopKF;
        }
    }
}

void Optimizer::LocalJointBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

//    cout << "No. Local Keyframes: " << lLocalKeyFrames.size() << endl;

    // Local MapPoints and MapObjects seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    list<MapObject*> lLocalMapObjects;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
            }
        }

        vector<MapObject*> vpMOs = (*lit)->GetMapObjectMatches();
        for (auto pMO : vpMOs)
        {
            if (pMO)
            {
                if (pMO->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapObjects.push_back(pMO);
                    pMO->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;
    std::set<long unsigned int> msKeyframeIDs;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        msKeyframeIDs.insert(pKFi->mnId);
        // cout << "KF ID: " << pKFi->mnId << endl;
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        msKeyframeIDs.insert(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices and edges
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    vector<EdgeSE3LieAlgebra*> vpEdgesCamObj;
    vector<KeyFrame*> vpEdgeKFCamObj;
    vector<MapObject*> vpMapObjectEdgeCamObj;

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    const float invSigmaObject = 1e3;
    const float thHuberObjectSquare = 1e3;
    const float thHuberObject = sqrt(thHuberObjectSquare);

    unsigned long maxMPid = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    if(pMP->mnId > maxMPid)
                        maxMPid = pMP->mnId;
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    if(pMP->mnId > maxMPid)
                        maxMPid = pMP->mnId;
                }
            }
        }
    }

    // Set map object vertices and edges
    for (auto pMO : lLocalMapObjects)
    {
        if (!pMO->isDynamic())
        {
            g2o::VertexSE3Expmap *vSE3Obj = new g2o::VertexSE3Expmap();
            vSE3Obj->setEstimate(Converter::toSE3Quat(pMO->SE3Tow));
            int id = pMO->mnId + maxKFid + maxMPid + 2;
            vSE3Obj->setId(id);
            optimizer.addVertex(vSE3Obj);

            const map<KeyFrame*, size_t> observations = pMO->GetObservations();

            for (auto observation : observations)
            {
                KeyFrame* pKFi = observation.first;
                if (msKeyframeIDs.count(pKFi->mnId) == 0)
                    continue;

                if(!pKFi->isBad())
                {
                    auto mvpObjectDetections = pKFi->GetObjectDetections();
                    // cout << "Object KF ID: " << pKFi->mnId << endl;
                    EdgeSE3LieAlgebra* e = new EdgeSE3LieAlgebra();
                    e->setVertex(0, optimizer.vertex(pKFi->mnId));
                    e->setVertex(1, optimizer.vertex(id));
                    auto det = mvpObjectDetections[observation.second];
                    e->setMeasurement(Converter::toSE3Quat(det->SE3Tco));
                    Eigen::Matrix<double, 6, 6> Info = Eigen::Matrix<double, 6, 6>::Identity();
                    Info*= invSigmaObject;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberObject);

                    optimizer.addEdge(e);
                    vpEdgesCamObj.push_back(e);
                    vpEdgeKFCamObj.push_back(pKFi);
                    vpMapObjectEdgeCamObj.push_back(pMO);
                }
            }
        }
    }

    if(pbStopFlag)
    {
        if(*pbStopFlag)
        {
            // cout << "Local BA hasn't finished, but abort signal triggered!!!!!!!!!!!!!!!!!" << endl;
            return;
        }
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
    {
        if(*pbStopFlag)
        {
            // cout << "Local BA hasn't finished, but abort signal triggered!!!!!!!!!!!!!!!!!" << endl;
            return;
        }
    }

    if(bDoMore)
    {
        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesCamObj.size(); i<iend;i++)
        {
            auto e = vpEdgesCamObj[i];

            if(e->chi2() > thHuberObjectSquare)
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
            // cout << "Edge " << vpEdgeKFCamObj[i]->mnId << "-" << vpMapObjectEdgeCamObj[i]->mnId << " Loss: " << e->chi2() << endl;
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    vector<pair<KeyFrame*, MapPoint*>> vToEraseCamPoints;
    vector<pair<KeyFrame*, MapObject*>> vToEraseCamObjects;
    vToEraseCamPoints.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
    vToEraseCamObjects.reserve(vpEdgesCamObj.size());

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToEraseCamPoints.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToEraseCamPoints.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesCamObj.size(); i<iend;i++)
    {
        auto e = vpEdgesCamObj[i];
        MapObject* pMO = vpMapObjectEdgeCamObj[i];

        if(e->chi2() > thHuberObjectSquare)
        {
            KeyFrame* pKFi = vpEdgeKFCamObj[i];
            vToEraseCamObjects.push_back(make_pair(pKFi, pMO));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToEraseCamPoints.empty())
    {
        for(size_t i=0; i<vToEraseCamPoints.size(); i++)
        {
            KeyFrame* pKFi = vToEraseCamPoints[i].first;
            MapPoint* pMPi = vToEraseCamPoints[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if (!vToEraseCamObjects.empty())
    {
        for(size_t i=0; i<vToEraseCamObjects.size(); i++)
        {
            KeyFrame* pKFi = vToEraseCamObjects[i].first;
            MapObject* pMO = vToEraseCamObjects[i].second;
            pKFi->EraseMapObjectMatch(pMO);
            pMO->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    //Objects
    for (auto pMO : lLocalMapObjects)
    {
        if (!pMO->isDynamic() && !pMO->isBad())
        {
            g2o::VertexSE3Expmap* vSE3Obj = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pMO->mnId + maxKFid + maxMPid + 2));
            g2o::SE3Quat SE3Tow = vSE3Obj->estimate();
            Eigen::Matrix4f SE3Two = Converter::toMatrix4f(SE3Tow).inverse();
            pMO->SetObjectPoseSE3(SE3Two);
        }
    }
    Optimizer::nBAdone++;

}

}