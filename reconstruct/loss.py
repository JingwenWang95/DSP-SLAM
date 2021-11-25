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

import torch
from reconstruct.loss_utils import decode_sdf, get_batch_sdf_jacobian, get_points_to_pose_jacobian_sim3, sdf_to_occupancy


def compute_sdf_loss(decoder, pts_surface_cam, t_obj_cam, latent_vector):
    """
    :param decoder: DeepSDF decoder
    :param pts_surface_cam: surface points under camera coordinate (N, 3)
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param latent_vector: shape code
    :return: Jacobian wrt pose (N, 1, 7), Jacobian wrt shape code (N, 1, code_len), error residuals (N, 1, 1)
    """
    # (n_sample_surface, 3)
    pts_surface_obj = \
        (pts_surface_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]

    res_sdf, de_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_surface_obj, 1)
    # SDF term Jacobian
    de_dxo = de_di[..., -3:]
    # Jacobian for pose
    dxo_dtoc = get_points_to_pose_jacobian_sim3(pts_surface_obj)
    jac_toc = torch.bmm(de_dxo, dxo_dtoc)
    # Jacobian for code
    jac_code = de_di[..., :-3]

    return jac_toc, jac_code, res_sdf


def compute_render_loss(decoder, ray_directions, depth_obs, t_obj_cam, sampled_ray_depth, latent_vector, th=0.01):
    """
    :param decoder: DeepSDF decoder
    :param ray_directions: (N, 3) under camera coordinate
    :param depth_obs: (N,) observed depth values for foreground pixels, 1.1 * d_max for background pixels
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :param sampled_ray_depth: (M,) linspace between d_min and d_max
    :param latent_vector: shape code
    :param th: cut-off threshold for converting SDF to occupancy
    :return: Jacobian wrt pose (K, 1, 7), Jacobian wrt shape code (K, 1, code_len), error residuals (K, 1, 1)
    Note that K is the number of points that have non-zero Jacobians out of a total of N * M points
    """

    # (n_rays, num_samples_per_ray, 3) = (n_rays, 1, 3) * (num_samples_per_ray, 1)
    sampled_points_cam = ray_directions[..., None, :] * sampled_ray_depth[:, None]
    # (n_rays, num_samples_per_ray, 3)
    sampled_points_obj = \
        (sampled_points_cam[..., None, :] * t_obj_cam.cuda()[:3, :3]).sum(-1) + t_obj_cam.cuda()[:3, 3]
    n_rays = sampled_points_obj.shape[0]
    n_depths = sampled_ray_depth.shape[0]

    # (num_rays, num_samples_per_ray)
    valid_indices = torch.where(torch.norm(sampled_points_obj, dim=-1) < 1.0)
    # (n_valid, 3) flattened = (n_rays, n_depth_sample, 3)[x, y, :]
    query_points_obj = sampled_points_obj[valid_indices[0], valid_indices[1], :]

    # if too few query points, return immediately
    if query_points_obj.shape[0] < 10:
        return None

    # flattened
    with torch.no_grad():
        sdf_values = decode_sdf(decoder, latent_vector, query_points_obj).squeeze()

    if sdf_values is None:
        raise Exception("no valid query points?")

    # Full dimension (n_rays, n_samples_per_ray)
    occ_values = torch.full((n_rays, n_depths), 0.).cuda()
    valid_indices_x, valid_indices_y = valid_indices  # (indices on x, y dimension)
    occ_values[valid_indices_x, valid_indices_y] = sdf_to_occupancy(sdf_values, th=th)

    with_grad = (sdf_values > -th) & (sdf_values < th)
    with_grad_indices_x = valid_indices_x[with_grad]
    with_grad_indices_y = valid_indices_y[with_grad]

    # point-wise values, i.e. multiple points might belong to one pixel (m, n_samples_per_ray)
    occ_values_with_grad = occ_values[with_grad_indices_x, :]
    m = occ_values_with_grad.shape[0]  # number of points with grad
    d_min = sampled_ray_depth[0]
    d_max = sampled_ray_depth[-1]

    # Render function
    acc_trans = torch.cumprod(1 - occ_values_with_grad, dim=-1)
    acc_trans_augment = torch.cat(
        (torch.ones(m, 1).cuda(), acc_trans),
        dim=-1
    )
    o = torch.cat(
        (occ_values_with_grad, torch.ones(m, 1).cuda()),
        dim=-1
    )
    d = torch.cat(
        (sampled_ray_depth, torch.tensor([1.1 * d_max]).cuda()),
        dim=-1
    )
    term_prob = (o * acc_trans_augment)
    # rendered depth values (m,)
    d_u = torch.sum(d * term_prob, dim=-1)
    var_u = torch.sum(term_prob * (d[None, :] - d_u[:, None]) ** 2, dim=-1)

    # Get Jacobian of depth residual wrt occupancy probability de_do
    o_k = occ_values[with_grad_indices_x, with_grad_indices_y]
    l = torch.arange(n_depths).cuda()
    l = l[None, :].repeat(m, 1)
    acc_trans[l < with_grad_indices_y[:, None]] = 0.
    de_do = acc_trans.sum(dim=-1) / (1. - o_k)

    # Remove points with zero gradients, and get de_ds = de_do * do_ds
    non_zero_grad = (de_do > 1e-2)
    de_do = de_do[non_zero_grad]
    d_u = d_u[non_zero_grad]
    delta_d = (d_max - d_min) / (n_depths - 1)
    do_ds = -1. / (2 * th)
    de_ds = (de_do * delta_d * do_ds).view(-1, 1, 1)

    # get residuals
    with_grad_indices_x = with_grad_indices_x[non_zero_grad]
    with_grad_indices_y = with_grad_indices_y[non_zero_grad]
    depth_obs_non_zero_grad = depth_obs[with_grad_indices_x]  # (m,)
    res_d = depth_obs_non_zero_grad - d_u  # (m,)

    # make it more robust and stable
    res_d[res_d > 0.30] = 0.30
    res_d[res_d < -0.30] = -0.30
    res_d = res_d.view(-1, 1, 1)

    pts_with_grad = sampled_points_obj[with_grad_indices_x, with_grad_indices_y]
    _, ds_di = get_batch_sdf_jacobian(decoder, latent_vector, pts_with_grad, 1)
    de_di = de_ds * ds_di  # (m, 1, code_len + 3)
    de_dxo = de_di[..., -3:]  # (m, 1, 3)
    # Jacobian for pose and code
    dxo_dtoc = get_points_to_pose_jacobian_sim3(pts_with_grad)
    jac_toc = torch.bmm(de_dxo, dxo_dtoc)
    jac_code = de_di[..., :-3]  # (m, 1, code_len)

    return jac_toc, jac_code, res_d


def compute_rotation_loss_sim3(t_obj_cam):
    """
    :param t_obj_cam: c2o transformation (4, 4) in Sim(3)
    :return: Jacobian and residual of rotation regularization term
    """
    # E_rot = 1 - ry * ng
    t_cam_obj = torch.inverse(t_obj_cam)
    r_co = t_cam_obj[:3, :3]
    scale = torch.det(r_co) ** (1 / 3)
    r_co /= scale
    r_oc = torch.inverse(r_co)

    ey = torch.tensor([0., 1., 0.])
    ng = torch.tensor([0., -1., 0.])
    ry = torch.mv(r_co, ey)
    res_rot = 1. - torch.dot(ry, ng)
    if res_rot < 1e-7:
        return torch.zeros(7), 0.

    J_rot = torch.cross(torch.mv(r_oc, ng), ey)
    J_sim3 = torch.zeros(7)
    J_sim3[3:6] = J_rot

    return J_sim3, res_rot
