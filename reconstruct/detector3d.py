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
import mmcv
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.apis import inference_detector, convert_SyncBN


def get_detector3d(configs):
    return Detector3D(configs)


class Detector3D(object):
    def __init__(self, configs):
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = configs.Detector3D.config_path
        checkpoint = configs.Detector3D.weight_path

        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')

        config.model.pretrained = None
        convert_SyncBN(config.model)
        config.model.train_cfg = None
        self.model = build_model(config.model, test_cfg=config.get('test_cfg'))

        if checkpoint is not None:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint['meta']:
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                self.model.CLASSES = config.class_names
            if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
                self.model.PALETTE = checkpoint['meta']['PALETTE']
        self.model.cfg = config  # save the config in the model for convenience
        self.model.to(device)
        self.model.eval()

    def make_prediction(self, velo_file):
        predictions, data = inference_detector(self.model, velo_file)
        # Car's label is 0 in KITTI dataset
        labels = predictions[0]["labels_3d"]
        scores = predictions[0]["scores_3d"]
        valid_mask = (labels == 0) & (scores > 0.0)
        boxes = predictions[0]["boxes_3d"].tensor

        return boxes[valid_mask]

