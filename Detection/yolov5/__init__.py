import os
import sys
import cv2

import torch
from .utils.utils import non_max_suppression, scale_coords
from .utils.datasets import letterbox
import numpy as np
from .models.yolo import Model
import yaml

# import sys
# sys.path.append('../yolov5_1')


__all__ = ['YOLOv5DETECTOR']


class YOLOv5DETECTOR:

        def __init__(self, model_cfg, inference_cfg):
                self.model_cfg = model_cfg
                self.inference_cfg = inference_cfg
                self.build_model()

        def build_model(self):

                weights = self.inference_cfg['weights']
                device = self.inference_cfg['device']

                ckpt = torch.load(weights, map_location=device)  # load checkpoint

                # check_ckpt = self.load_ckpt(ckpt)
                # print(check_ckpt.keys())

                self.model = Model(self.model_cfg, ch=3, nc=self.model_cfg['nc']).to(device)  # create
                
                self.model.load_state_dict(ckpt,strict=True)  # load

                self.model.eval()

                # self.model = torch.load(weights, map_location=device)['model']
                # self.model.to(device).eval()

        @torch.no_grad()
        def load_ckpt(self,ckpt):
                new_state = {}
                for ckpt_key in ckpt:
                        if ckpt_key.startswith('model.model'):
                                new_state[ckpt_key[6:]] = ckpt[ckpt_key]
                        else:
                                new_state[ckpt_key] = ckpt[ckpt_key]
                return new_state

        @torch.no_grad()
        def predict_box(self, img0, plot_box=False):
                # 推理配置参数
                imagesize = self.inference_cfg['imagesize']
                conf_thres = self.inference_cfg['conf_thres']
                max_det = self.inference_cfg['max_det']
                classes = self.inference_cfg['classes']
                iou_thres = self.inference_cfg['iou_thres']
                agnostic_nms = self.inference_cfg['agnostic_nms']
                half = self.inference_cfg['half']
                device = self.inference_cfg['device']
                weights = self.inference_cfg['weights']

                # 预处理
                im = letterbox(img0, new_shape=(imagesize, imagesize))[0]
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im)
                im = im.astype(np.float32)
                
                im = torch.from_numpy(im).to(device)
                im.float()
                im /= 255.0
                if im.ndimension() == 3:
                        im = im.unsqueeze(0)
                pred = self.model(im.float())[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres,True, classes, agnostic_nms)
                det = pred[0]  # Only  need first image result
                if det is not None and len(det):
                        # print(det)
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                return det
