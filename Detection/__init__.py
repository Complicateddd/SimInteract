from .yolov5 import YOLOv5DETECTOR
from .Sort import Sort

Detlib = {'yolov5': YOLOv5DETECTOR }


__all__ = ['build_detector','Sort']


def build_detector(model_name ,model_cfg, inference_cfg):

        detector_choose = Detlib[model_name]

        return detector_choose(model_cfg, inference_cfg)
