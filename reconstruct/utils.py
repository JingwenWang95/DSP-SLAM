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
import numpy as np
import json
from addict import Dict
import plyfile
import skimage.measure as measure
from deep_sdf.workspace import config_decoder

# colors used for visualization
color_table = [[230. / 255., 0., 0.],  # red
               [60. / 255., 180. / 255., 75. / 255.],  # green
               [0., 0., 255. / 255.],  # blue
               [255. / 255., 0, 255. / 255.],
               [255. / 255., 165. / 255., 0.],
               [128. / 255., 0, 128. / 255.],
               [0., 255. / 255., 255. / 255.],
               [210. / 255., 245. / 255., 60. / 255.],
               [250. / 255., 190. / 255., 190. / 255.],
               [0., 128. / 255., 128. / 255.]
               ]


def set_view(vis, dist=100., theta=np.pi/6.):
    """
    :param vis: o3d visualizer
    :param dist: eye-to-world distance, assume eye is looking at world origin
    :param theta: tilt-angle around x-axis of world coordinate
    """
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # world to eye
    T = np.array([[1., 0., 0., 0.],
                  [0., np.cos(theta), -np.sin(theta), 0.],
                  [0., np.sin(theta), np.cos(theta), dist],
                  [0., 0., 0., 1.]])

    cam.extrinsic = T
    vis_ctr.convert_from_pinhole_camera_parameters(cam)


def read_calib_file(filepath):
    """Read in a KITTI calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def get_configs(cfg_file):
    with open(cfg_file) as f:
        cfg_dict = json.load(f)
    return ForceKeyErrorDict(**cfg_dict)


def get_decoder(configs):
    return config_decoder(configs.DeepSDF_DIR)


def create_voxel_grid(vol_dim=128):
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (vol_dim - 1)

    overall_index = torch.arange(0, vol_dim ** 3, 1, out=torch.LongTensor())
    values = torch.zeros(vol_dim ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    values[:, 2] = overall_index % vol_dim
    values[:, 1] = (overall_index.long() / vol_dim) % vol_dim
    values[:, 0] = ((overall_index.long() / vol_dim) / vol_dim) % vol_dim

    # transform first 3 columns
    # to be the x, y, z coordinate
    values[:, 0] = (values[:, 0] * voxel_size) + voxel_origin[2]
    values[:, 1] = (values[:, 1] * voxel_size) + voxel_origin[1]
    values[:, 2] = (values[:, 2] * voxel_size) + voxel_origin[0]

    return values


def convert_sdf_voxels_to_mesh(pytorch_3d_sdf_tensor):
    """
    Convert sdf samples to mesh
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :return vertices and faces of the mesh
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().detach().numpy()
    voxels_dim = numpy_3d_sdf_tensor.shape[0]
    voxel_size = 2.0 / (voxels_dim - 1)
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    voxel_grid_origin = np.array([-1., -1., -1.])
    verts[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    verts[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    verts[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    return verts, faces


def write_mesh_to_ply(v, f, ply_filename_out):
    # try writing to the ply file

    num_verts = v.shape[0]
    num_faces = f.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(v[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((f[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
