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

import numpy as np
import os
import cv2
import torch
from reconstruct.loss_utils import get_rays, get_time
from reconstruct.utils import ForceKeyErrorDict, read_calib_file, load_velo_scan
from reconstruct import get_detectors


class FrameWithLiDAR:
    def __init__(self, sequence, frame_id):
        # Load sequence properties
        self.configs = sequence.configs
        self.rgb_dir = sequence.rgb_dir
        self.velo_dir = sequence.velo_dir
        self.lbl2d_dir = sequence.lbl2d_dir
        self.lbl3d_dir = sequence.lbl3d_dir
        self.K = sequence.K_cam
        self.invK = sequence.invK_cam
        self.T_cam_velo = sequence.T_cam_velo
        self.online = sequence.online
        self.detector_2d = sequence.detector_2d
        self.detector_3d = sequence.detector_3d
        self.max_lidar_pts = self.configs.num_lidar_max
        self.min_lidar_pts = self.configs.num_lidar_min
        self.min_mask_area = self.configs.min_mask_area

        # Load image and LiDAR measurements
        self.frame_id = frame_id
        rgb_file = os.path.join(self.rgb_dir, "{:06d}".format(frame_id) + ".png")
        self.velo_file = os.path.join(self.velo_dir, "{:06d}".format(frame_id) + ".bin")
        self.img_bgr = cv2.imread(rgb_file)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = self.img_rgb.shape
        self.velo_pts = load_velo_scan(self.velo_file)
        self.instances = []

    def get_colored_pts(self):
        velo_pts_cam = (self.velo_pts[:, None, :3] * self.T_cam_velo[:3, :3]).sum(-1) + self.T_cam_velo[:3, 3]
        velo_pts_cam = velo_pts_cam[(velo_pts_cam[:, 2] > 0), :]

        img_h, img_w, _ = self.img_rgb.shape

        uv_hom = (velo_pts_cam[:, None, :] * self.K).sum(-1)
        uv = uv_hom[:, :2] / uv_hom[:, 2, None]
        in_fov = (uv[:, 0] > 0) & (uv[:, 0] < img_w) & (uv[:, 1] > 0) & (uv[:, 1] < img_h)
        uv = uv[in_fov].astype(np.int32)
        velo_pts_cam_fov = velo_pts_cam[in_fov].astype(np.float32)
        colors_fov = self.img_rgb[uv[:, 1], uv[:, 0], :] / 255.

        return velo_pts_cam_fov, colors_fov

    def pixels_sampler(self, bbox_2d, mask):
        alpha = int(self.configs.downsample_ratio)
        expand_len = 5
        max_w, max_h = self.img_w - 1, self.img_h - 1
        # Expand the crop such that it will not be too tight
        l, t, r, b = list(bbox_2d.astype(np.int32))
        l = l - 5 if l > expand_len else 0
        t = t - 5 if t > expand_len else 0
        r = r + 5 if r < max_w - expand_len else max_w
        b = b + 5 if b < max_h - expand_len else max_h
        # Sample pixels inside the 2d box
        crop_H, crop_W = b - t + 1, r - l + 1
        hh = np.linspace(t, b, int(crop_H / alpha)).astype(np.int32)
        ww = np.linspace(l, r, int(crop_W / alpha)).astype(np.int32)
        crop_h, crop_w = hh.shape[0], ww.shape[0]
        hh = hh[:, None].repeat(crop_w, axis=1)
        ww = ww[None, :].repeat(crop_h, axis=0)
        sampled_pixels = np.concatenate([hh[:, :, None], ww[:, :, None]], axis=-1).reshape(-1, 2)
        vv, uu = sampled_pixels[:, 0], sampled_pixels[:, 1]
        non_surface = ~mask[vv, uu]
        sampled_pixels_non_surface = np.concatenate([uu[non_surface, None], vv[non_surface, None]], axis=-1)

        return sampled_pixels_non_surface

    def get_labels(self):
        labels_3d = self.detector_3d.make_prediction(self.velo_file).cpu().numpy()
        labels_2d = self.detector_2d.make_prediction(self.img_bgr)
        return labels_2d, labels_3d

    def get_detections(self):
        # Get 3D Detection first
        t1 = get_time()
        # get lidar points here
        if self.online:
            detections_3d = self.detector_3d.make_prediction(self.velo_file).cpu().numpy()
        else:
            label_path_3d = os.path.join(self.lbl3d_dir,  "%06d.lbl" % self.frame_id)
            detections_3d = torch.load(label_path_3d)
        t2 = get_time()
        print("3D detector takes %f seconds" % (t2 - t1))

        # sort according to depth order
        depth_order = np.argsort(detections_3d[:, 0])
        detections_3d = detections_3d[depth_order, :]
        for n in range(detections_3d.shape[0]):
            det_3d = detections_3d[n, :]
            trans, size, theta = det_3d[:3], det_3d[3:6], det_3d[6]
            # Get SE(3) transformation matrix from trans and theta
            T_velo_obj = np.array([[np.cos(theta), 0, -np.sin(theta), trans[0]],
                                   [-np.sin(theta), 0, -np.cos(theta), trans[1]],
                                   [0, 1, 0, trans[2] + size[2] / 2],
                                   [0, 0, 0, 1]]).astype(np.float32)
            T_obj_velo = np.linalg.inv(T_velo_obj)
            x, y, z = list(trans)
            # Filter out points that are too far away from car centroid, with radius 3.0 meters
            r = 3.0
            nearby = (self.velo_pts[:, 0] > x - r) & (self.velo_pts[:, 0] < x + r) & \
                     (self.velo_pts[:, 1] > y - r) & (self.velo_pts[:, 1] < y + r) & \
                     (self.velo_pts[:, 2] > z - r) & (self.velo_pts[:, 2] < z + r)
            points_nearby = self.velo_pts[nearby]
            points_obj = (points_nearby[:, None, :3] * T_obj_velo[:3, :3]).sum(-1) + T_obj_velo[:3, 3]
            # Further filter out the points that are outside the 3D bounding box
            w, l, h = list(size / 2)
            w *= 1.1
            l *= 1.1
            on_surface = (points_obj[:, 0] > -w) & (points_obj[:, 0] < w) & \
                         (points_obj[:, 1] > -h) & (points_obj[:, 1] < h) & \
                         (points_obj[:, 2] > -l) & (points_obj[:, 2] < l)
            pts_surface_velo = points_nearby[on_surface]
            # Sample from all the depth measurement
            N = pts_surface_velo.shape[0]
            if N > self.max_lidar_pts:
                sample_ind = np.linspace(0, N-1, self.max_lidar_pts).astype(np.int32)
                pts_surface_velo = pts_surface_velo[sample_ind, :]
            pts_surface_cam = (pts_surface_velo[:, None, :3] * self.T_cam_velo[:3, :3]).sum(-1) + self.T_cam_velo[:3, 3]
            T_cam_obj = self.T_cam_velo @ T_velo_obj
            T_cam_obj[:3, :3] *= l

            # Initialize detected instance
            instance = ForceKeyErrorDict()
            instance.T_cam_obj = T_cam_obj
            instance.scale = size
            instance.surface_points = pts_surface_cam.astype(np.float32)
            instance.num_surface_points = pts_surface_cam.shape[0]
            instance.is_front = T_cam_obj[2, 3] > 0.0
            instance.rays = None

            self.instances += [instance]

        # Get 2D Detection and associate with 3D instances
        t3 = get_time()
        if self.online:
            det_2d = self.detector_2d.make_prediction(self.img_bgr)
        else:
            label_path2d = os.path.join(self.lbl2d_dir, "%06d.lbl" % self.frame_id)
            det_2d = torch.load(label_path2d)
        t4 = get_time()
        print("2D detctor takes %f seconds" % (t4 - t3))

        img_h, img_w, _ = self.img_rgb.shape
        masks_2d = det_2d["pred_masks"]
        bboxes_2d = det_2d["pred_boxes"]

        # If no 2D detections, return right away
        if masks_2d.shape[0] == 0:
            return

        # Occlusion masks
        occ_mask = np.full([img_h, img_w], False, dtype=np.bool)
        prev_mask = None
        for instance in self.instances:
            if not instance.is_front:
                continue
            # Project LiDAR points to image plane
            surface_points = instance.surface_points
            pixels_homo = (surface_points[:, None, :] * self.K).sum(-1)
            pixels_uv = (pixels_homo[:, :2] / pixels_homo[:, 2, None])
            in_fov = (pixels_uv[:, 0] > 0) & (pixels_uv[:, 0] < img_w) & \
                     (pixels_uv[:, 1] > 0) & (pixels_uv[:, 1] < img_h)
            pixels_coord = pixels_uv[in_fov].astype(np.int32)
            # Check all the n 2D masks, and see how many projected points are inside them
            points_in_masks = [masks_2d[n, pixels_coord[:, 1], pixels_coord[:, 0]] for n in range(masks_2d.shape[0])]
            num_matches = np.array([points_in_mask[points_in_mask].shape[0] for points_in_mask in points_in_masks])
            max_num_matchess = num_matches.max()

            if max_num_matchess > pixels_coord.shape[0] * 0.5:
                n = np.argmax(num_matches)
                instance.mask = masks_2d[n, ...]
                instance.bbox = bboxes_2d[n, ...]

                if instance.mask[instance.mask].shape[0] > self.min_mask_area:
                    # Sample non-surface pixels
                    non_surface_pixels = self.pixels_sampler(instance.bbox, instance.mask)
                    if non_surface_pixels.shape[0] > 200:
                        sample_ind = np.linspace(0, non_surface_pixels.shape[0]-1, 200).astype(np.int32)
                        non_surface_pixels = non_surface_pixels[sample_ind, :]

                    pixels_inside_bb = np.concatenate([pixels_uv, non_surface_pixels], axis=0)
                    # rays contains all, but depth should only contain foreground
                    instance.rays = get_rays(pixels_inside_bb, self.invK).astype(np.float32)
                    instance.depth = surface_points[:, 2].astype(np.float32)

                # Create occlusion mask
                if prev_mask is not None:
                    occ_mask = occ_mask | prev_mask
                instance.occ_mask = occ_mask
                prev_mask = masks_2d[n, ...]


class KITIISequence:
    def __init__(self, data_dir, configs):
        self.root_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "image_2")
        self.velo_dir = os.path.join(data_dir, "velodyne")
        self.calib_file = os.path.join(data_dir, "calib.txt")
        self.load_calib()
        self.num_frames = len(os.listdir(self.rgb_dir))
        self.configs = configs
        self.online = self.configs.detect_online
        # Pre-stored label path
        self.lbl2d_dir = self.configs.path_label_2d
        self.lbl3d_dir = self.configs.path_label_3d
        if not self.online:
            assert self.lbl2d_dir is not None, print()
            assert self.lbl3d_dir is not None, print()
        # Detectors
        self.detector_2d, self.detector_3d = get_detectors(self.configs)
        self.current_frame = None
        self.detections_in_current_frame = None

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # Load the calibration file
        filedata = read_calib_file(self.calib_file)

        # Load projection matrix P_cam2_cam0, and compute perspective instrinsics K of cam2
        P_cam2_cam0 = np.reshape(filedata['P2'], (3, 4))
        self.K_cam = P_cam2_cam0[0:3, 0:3].astype(np.float32)
        self.invK_cam = np.linalg.inv(self.K_cam).astype(np.float32)

        # Load the transfomration from T_cam0_velo, and compute the transformation T_cam2_velo
        T_cam0_velo, T_cam2_cam0 = np.eye(4), np.eye(4)
        T_cam0_velo[:3, :] = np.reshape(filedata['Tr'], (3, 4))
        T_cam2_cam0[0, 3] = P_cam2_cam0[0, 3] / P_cam2_cam0[0, 0]
        self.T_cam_velo = T_cam2_cam0.dot(T_cam0_velo).astype(np.float32)

    def get_frame_by_id(self, frame_id):
        self.current_frame = FrameWithLiDAR(self, frame_id)
        self.current_frame.get_detections()
        self.detections_in_current_frame = self.current_frame.instances
        return self.detections_in_current_frame

    def get_labels_and_save(self):
        if not os.path.exists(self.lbl2d_dir):
            os.makedirs(self.lbl2d_dir)
        if not os.path.exists(self.lbl3d_dir):
            os.makedirs(self.lbl3d_dir)

        for frame_id in range(0, self.num_frames):
            frame = FrameWithLiDAR(self, frame_id)
            labels_2d, labels_3d = frame.get_labels()
            torch.save(labels_2d, os.path.join(self.lbl2d_dir, "%06d.lbl" % frame_id))
            torch.save(labels_3d, os.path.join(self.lbl3d_dir, "%06d.lbl" % frame_id))
            print("Finished saving frame %d" % frame_id)
