# Matt Beveridge, 2021

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class BluePriusDataset(MonoDataset):
    """
    Superclass for different types of Devins dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(BluePriusDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.full_res_shape = (1920, 1208)
        cx = 982.06345972
        cy = 617.04862261
        fx = 942.21001829
        fy = 942.91483399
        h, w = self.full_res_shape
        self.K = np.array([[fx/w, 0,    cx/w, 0],
                           [0,    fy/h, cy/h, 0],
                           [0,    0,    1,    0],
                           [0,    0,    0,    1]], dtype=np.float32)

        self.side_map = {
            "c":"camera_front", 
            "l":"camera_left",  
            "r":"camera_right",
            "wl":"camera_wing_left",
            "wr":"camera_wing_right"
        }


    def check_depth(self):
        return False


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class BluePriusRawDataset(BluePriusDataset):
    """
    Blue Prius dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(BluePriusRawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, f_str)
        return image_path
