from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os

import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

config_file = "../configs/my_mouth_test_e2e_mask_rcnn_R_50_C4_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE","cuda"])




model=build_detection_model(cfg).to("cpu")
loadedModel=torch.load("D:/XM/codes1230/maskrcnn-benchmark/weights/C4/model_final.pth")
model.load_state_dict(loadedModel.pop("model"))
torch.save(model.state_dict(),"model_final_for_windows1.pth", _use_new_zipfile_serialization=False)

# state_dict1 = torch.load("D:/XM/codes1230/maskrcnn-benchmark/weights/C4/model_final.pth")
# torch.save(state_dict1, "model_final_for_windows1.pth", _use_new_zipfile_serialization=False)

# model_state_dict = model.state_dict()
# loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
# align_and_update_state_dicts(model_state_dict,loaded_state_dict)
#
# # use strict loading
# model.load_state_dict(model_state_dict)