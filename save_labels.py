from reconstruct.kitti_sequence import KITIISequence
from reconstruct.utils import get_configs


if __name__ == "__main__":
    config = "configs/config_kitti.json"
    config = get_configs(config)
    data_dir = "/media/jingwen/Data/kitti_odom/dataset/sequences/07"
    sequence = KITIISequence(data_dir, config)
    sequence.get_labels_and_save()