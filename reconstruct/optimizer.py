#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import math
import numpy as np
import torch
from reconstruct.utils import ForceKeyErrorDict, create_voxel_grid, convert_sdf_voxels_to_mesh
from reconstruct.loss import compute_sdf_loss, compute_render_loss, compute_rotation_loss_sim3
from reconstruct.loss_utils import decode_sdf, get_robust_res, exp_se3, exp_sim3, get_time


class Optimizer(object):
    def __init__(self, decoder, configs):
        self.decoder = decoder
        optim_cfg = configs.optimizer
        self.k1 = optim_cfg.joint_optim.k1
        self.k2 = optim_cfg.joint_optim.k2
        self.k3 = optim_cfg.joint_optim.k3
        self.k4 = optim_cfg.joint_optim.k4
        self.b1 = optim_cfg.joint_optim.b1
        self.b2 = optim_cfg.joint_optim.b2
        self.lr = optim_cfg.joint_optim.learning_rate
        self.s_damp = optim_cfg.joint_optim.scale_damping
        self.num_iterations_joint_optim = optim_cfg.joint_optim.num_iterations
        self.code_len = optim_cfg.code_len
        self.num_depth_samples = optim_cfg.num_depth_samples
        self.cut_off = optim_cfg.cut_off_threshold
        if configs.data_type == "KITTI":
            self.num_iterations_pose_only = optim_cfg.pose_only_optim.num_iterations

    def estimate_pose_cam_obj(self, t_co_se3, scale, pts, code):
        """
        :param t_co_se3: o2c transformation (4, 4) in SE(3)
        :param scale: object scale
        :param pts: surface points (M, 3)
        :param code: shape code
        :return: optimized o2c transformation
        """
        t_cam_obj = torch.from_numpy(t_co_se3)
        t_cam_obj[:3, :3] *= scale
        t_obj_cam = torch.inverse(t_cam_obj)
        latent_vector = torch.from_numpy(code).cuda()
        pts_surface = torch.from_numpy(pts).cuda()

        for e in range(self.num_iterations_pose_only):
            start = get_time()
            # 1. Compute SDF (3D) loss
            de_dsim3_sdf, de_dc_sdf, res_sdf = \
                compute_sdf_loss(self.decoder, pts_surface,
                                      t_obj_cam,
                                      latent_vector)
            _, sdf_loss, _ = get_robust_res(res_sdf, 0.05)

            j_sdf = de_dsim3_sdf[..., :6]
            hess = torch.bmm(j_sdf.transpose(-2, -1), j_sdf).sum(0).squeeze().cpu() / j_sdf.shape[0]
            hess += 1e-2 * torch.eye(6)
            b = -torch.bmm(j_sdf.transpose(-2, -1), res_sdf).sum(0).squeeze().cpu() / j_sdf.shape[0]
            dx = torch.mv(torch.inverse(hess), b)
            delta_t = exp_se3(dx)
            t_obj_cam = torch.mm(delta_t, t_obj_cam)

            if e == 4:
                inliers_mask = torch.abs(res_sdf).squeeze() <= 0.05
                pts_surface = pts_surface[inliers_mask, :]

            # print("Object pose-only optimization: Iter %d, sdf loss: %f" % (e, sdf_loss))

        # Convert back to SE3
        t_cam_obj = torch.inverse(t_obj_cam)
        t_cam_obj[:3, :3] /= scale

        return t_cam_obj

    def reconstruct_object(self, t_cam_obj, pts, rays, depth, code=None):
        """
        :param t_cam_obj: object pose, object-to-camera transformation
        :param pts: surface points, under camera coordinate (M, 3)
        :param rays: sampled ray directions (N, 3)
        :param depth: depth values (K,) only contain foreground pixels, K = M for KITTI
        :return: optimized opject pose and shape, saved as a dict
        """
        # Always start from zero code
        if code is None:
            latent_vector = torch.zeros(self.code_len).cuda()
        else:
            latent_vector = torch.from_numpy(code[:self.code_len]).cuda()

        # Initial Pose Estimate
        t_cam_obj = torch.from_numpy(t_cam_obj)
        t_obj_cam = torch.inverse(t_cam_obj)
        # ray directions within Omega_r
        ray_directions = torch.from_numpy(rays).cuda()
        # depth observations within Omega_r
        n_foreground_rays = depth.shape[0]
        n_background_rays = rays.shape[0] - n_foreground_rays
        # print("rays: %d, total rays: %d" % (n_foreground_rays, n_background_rays))
        depth_obs = np.concatenate([depth, np.zeros(n_background_rays)], axis=0).astype(np.float32)
        depth_obs = torch.from_numpy(depth_obs).cuda()
        # surface points within Omega_s
        pts_surface = torch.from_numpy(pts).cuda()

        start = get_time()
        loss = 0.
        for e in range(self.num_iterations_joint_optim):
            # get depth range and sample points along the rays
            t_cam_obj = torch.inverse(t_obj_cam)
            scale = torch.det(t_cam_obj[:3, :3]) ** (1 / 3)
            # print("Scale: %f" % scale)
            depth_min, depth_max = t_cam_obj[2, 3] - 1.0 * scale, t_cam_obj[2, 3] + 1.0 * scale
            sampled_depth_along_rays = torch.linspace(depth_min, depth_max, self.num_depth_samples).cuda()
            # set background depth to d'
            depth_obs[n_foreground_rays:] = 1.1 * depth_max

            # 1. Compute SDF (3D) loss
            sdf_rst = compute_sdf_loss(self.decoder, pts_surface, t_obj_cam, latent_vector)
            if sdf_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_sdf, de_dc_sdf, res_sdf = sdf_rst
            robust_res_sdf, sdf_loss, _ = get_robust_res(res_sdf, self.b2)
            if math.isnan(sdf_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            # 2. Compute Render (2D) Loss
            render_rst = compute_render_loss(self.decoder, ray_directions, depth_obs, t_obj_cam,
                                             sampled_depth_along_rays, latent_vector, th=self.cut_off)
            # in case rendering fails
            if render_rst is None:
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)
            else:
                de_dsim3_render, de_dc_render, res_render = render_rst

            # print("rays gradients on python side: %d" % de_dsim3_render.shape[0])
            robust_res_render, render_loss, _ = get_robust_res(res_render, self.b1)
            if math.isnan(render_loss):
                return ForceKeyErrorDict(t_cam_obj=None, code=None, is_good=False, loss=loss)

            # 3. Rotation prior
            drot_dsim3, res_rot = compute_rotation_loss_sim3(t_obj_cam)

            loss = self.k1 * render_loss + self.k2 * sdf_loss
            z = latent_vector.cpu()

            # Compute Jacobian and Hessia
            pose_dim = 7

            J_sdf = torch.cat([de_dsim3_sdf, de_dc_sdf], dim=-1)
            H_sdf = self.k2 * torch.bmm(J_sdf.transpose(-2, -1), J_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]
            b_sdf = -self.k2 * torch.bmm(J_sdf.transpose(-2, -1), robust_res_sdf).sum(0).squeeze().cpu() / J_sdf.shape[0]

            J_render = torch.cat([de_dsim3_render, de_dc_render], dim=-1)
            H_render = self.k1 * torch.bmm(J_render.transpose(-2, -1), J_render).sum(0).squeeze().cpu() / J_render.shape[0]
            b_render = -self.k1 * torch.bmm(J_render.transpose(-2, -1), robust_res_render).sum(0).squeeze().cpu() / J_render.shape[0]

            H = H_render + H_sdf
            H[pose_dim:pose_dim + self.code_len, pose_dim:pose_dim + self.code_len] += self.k3 * torch.eye(self.code_len)
            b = b_render + b_sdf
            b[pose_dim:pose_dim + self.code_len] -= self.k3 * z

            # Rotation regularization
            drot_dsim3 = drot_dsim3.unsqueeze(0)
            H_rot = torch.mm(drot_dsim3.transpose(-2, -1), drot_dsim3)
            b_rot = -(drot_dsim3.transpose(-2, -1) * res_rot).squeeze()
            H[:pose_dim, :pose_dim] += self.k4 * H_rot
            b[:pose_dim] -= self.k4 * b_rot
            # rot_loss = res_rot

            # add a small damping to the pose part
            H[:pose_dim, :pose_dim] += 1e0 * torch.eye(pose_dim)
            H[pose_dim-1, pose_dim-1] += self.s_damp  # add a large damping for scale
            # solve for the update vector
            dx = torch.mv(torch.inverse(H), b)
            delta_p = dx[:pose_dim]

            delta_c = dx[pose_dim:pose_dim + self.code_len]
            delta_t = exp_sim3(self.lr * delta_p)
            t_obj_cam = torch.mm(delta_t, t_obj_cam)
            latent_vector += self.lr * delta_c.cuda()

            # print("Object joint optimization: Iter %d, loss: %f, sdf loss: %f, "
            #       "render loss: %f, rotation loss: %f"
            #       % (e, loss, sdf_loss, render_loss, rot_loss))

        end = get_time()
        print("Reconstruction takes %f seconds" % (end - start))
        t_cam_obj = torch.inverse(t_obj_cam)
        return ForceKeyErrorDict(t_cam_obj=t_cam_obj.numpy(),
                                 code=latent_vector.cpu().numpy(),
                                 is_good=True, loss=loss)


class MeshExtractor(object):
    def __init__(self, decoder, code_len=64, voxels_dim=64):
        self.decoder = decoder
        self.code_len = code_len
        self.voxels_dim = voxels_dim
        with torch.no_grad():
            self.voxel_points = create_voxel_grid(vol_dim=self.voxels_dim).cuda()

    def extract_mesh_from_code(self, code):
        start = get_time()
        latent_vector = torch.from_numpy(code[:self.code_len]).cuda()
        sdf_tensor = decode_sdf(self.decoder, latent_vector, self.voxel_points)
        vertices, faces = convert_sdf_voxels_to_mesh(sdf_tensor.view(self.voxels_dim, self.voxels_dim, self.voxels_dim))
        vertices = vertices.astype("float32")
        faces = faces.astype("int32")
        end = get_time()
        print("Extract mesh takes %f seconds" % (end - start))
        return ForceKeyErrorDict(vertices=vertices, faces=faces)