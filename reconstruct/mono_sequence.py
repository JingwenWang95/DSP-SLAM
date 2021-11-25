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
from reconstruct.utils import ForceKeyErrorDict
from reconstruct import get_detectors


class Frame:
    def __init__(self, sequence, frame_id):
        # Load sequence properties
        self.configs = sequence.configs
        self.rgb_dir = sequence.rgb_dir
        self.lbl2d_dir = sequence.lbl2d_dir
        self.K = sequence.K_cam
        self.invK = sequence.invK_cam
        self.k1 = sequence.k1
        self.k2 = sequence.k2
        self.online = sequence.online
        self.detector_2d = sequence.detector_2d
        self.min_mask_area = self.configs.min_mask_area
        if sequence.data_type == "Redwood":
            self.object_class = "chairs"
        elif sequence.data_type == "Freiburg":
            self.object_class = "cars"
        self.frame_id = frame_id
        rgb_file = os.path.join(self.rgb_dir, "{:06d}".format(frame_id) + ".png")
        self.img_bgr = cv2.imread(rgb_file)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = self.img_rgb.shape
        self.instances = []

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

    def get_detections(self):
        # Get 2D Detection and background rays
        t1 = get_time()
        if self.online:
            det_2d = self.detector_2d.make_prediction(self.img_bgr, object_class=self.object_class)
        else:
            label_path2d = os.path.join(self.lbl2d_dir, "%06d.lbl" % self.frame_id)
            det_2d = torch.load(label_path2d)
        t2 = get_time()
        print("2D detctor takes %f seconds" % (t2 - t1))

        img_h, img_w, _ = self.img_rgb.shape
        masks_2d = det_2d["pred_masks"]
        bboxes_2d = det_2d["pred_boxes"]

        # If no 2D detections, return right away
        if masks_2d.shape[0] == 0:
            return

        # For redwood and freiburg cars, we only focus on the object in the middle
        max_id = np.argmax(masks_2d.sum(axis=-1).sum(axis=-1))
        mask_max = masks_2d[max_id, ...].astype(np.float32) * 255.
        bbox_max = bboxes_2d[max_id, ...]

        non_surface_pixels = self.pixels_sampler(bbox_max, mask_max.astype(np.bool8))

        if non_surface_pixels.shape[0] > 200:
            sample_ind = np.linspace(0, non_surface_pixels.shape[0]-1, 200).astype(np.int32)
            non_surface_pixels = non_surface_pixels[sample_ind, :]

        distortion_coef = np.array([self.k1, self.k2, 0.0, 0.0, 0.0])
        non_surface_pixels_undistort = cv2.undistortPoints(non_surface_pixels.reshape(1, -1, 2).astype(np.float32), self.K, distortion_coef, P=self.K).squeeze()
        background_rays_undist = get_rays(non_surface_pixels_undistort, self.invK).astype(np.float32)

        instance = ForceKeyErrorDict()
        instance.bbox = bbox_max
        instance.mask = mask_max
        instance.background_rays = background_rays_undist

        self.instances = [instance]


class MonoSequence:
    def __init__(self, data_dir, configs):
        self.root_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "image_0")

        # Get camera intrinsics
        fs = cv2.FileStorage(configs.slam_config_path, cv2.FILE_STORAGE_READ)
        fx = fs.getNode("Camera.fx").real()
        fy = fs.getNode("Camera.fy").real()
        cx = fs.getNode("Camera.cx").real()
        cy = fs.getNode("Camera.cy").real()
        k1 = fs.getNode("Camera.k1").real()
        k2 = fs.getNode("Camera.k2").real()
        self.K_cam = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.invK_cam = np.linalg.inv(self.K_cam)
        self.k1 = k1
        self.k2 = k2

        self.configs = configs
        self.data_type = self.configs.data_type
        assert self.data_type == "Redwood" or self.data_type == "Freiburg", print("Wrong data type, supported: Redwood and Freiburg")
        self.online = self.configs.detect_online
        # Pre-stored label path
        self.lbl2d_dir = self.configs.path_label_2d
        if not self.online:
            assert self.lbl2d_dir is not None, print()

        # Detectors
        self.detector_2d = get_detectors(self.configs)
        self.current_frame = None
        self.detections_in_current_frame = None

    def get_frame_by_id(self, frame_id):
        self.current_frame = Frame(self, frame_id)
        self.current_frame.get_detections()
        self.detections_in_current_frame = self.current_frame.instances
        return self.detections_in_current_frame

