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

import os
import numpy as np
import argparse
from reconstruct.optimizer import MeshExtractor
from reconstruct.utils import get_configs, get_decoder, write_mesh_to_ply


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-m', '--map_dir', type=str, required=True, help='path to map directory')
    parser.add_argument('-n', '--voxels_dim', type=int, default=128, help='voxels resolution for running marching cube')
    return parser


if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()
    decoder = get_decoder(get_configs(args.config))
    configs = get_configs(args.config)
    mesh_extractor = MeshExtractor(decoder, configs.optimizer.code_len, args.voxels_dim)

    map_dir = args.map_dir
    save_dir = os.path.join(map_dir, "objects")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(map_dir, "MapObjects.txt")) as f:
        lines = f.readlines()
        N = int(len(lines) / 3)
        for i in range(N):
            obj_id = int(lines[3 * i])
            line_pose = lines[3 * i + 1]
            pose = np.asarray([float(x) for x in line_pose.strip().split(" ")]).reshape(3, 4)
            pose = np.concatenate([pose, np.array([0., 0., 0., 1.]).reshape(1, 4)], axis=0)
            np.save(os.path.join(save_dir, "%d.npy" % obj_id), pose)
            code = []
            line_code = lines[3 * i + 2]
            for item in line_code.strip().split(" "):
                if len(item) > 0:
                    code += [float(item)]

            code = np.asarray(code).astype(np.float32)
            mesh = mesh_extractor.extract_mesh_from_code(code)
            write_mesh_to_ply(mesh.vertices, mesh.faces, os.path.join(save_dir, "%d.ply" % obj_id))
