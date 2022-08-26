import sys
import os

import torch

# sys.path.append("../frcnn")

# from frcnn import FRCNN

sys.path.append("./mmdetection")

# from mmdet import __version__
# from mmdet.apis import init_detector


def load_yolo(model_type='yolov3_spp', device="cuda:0"):
    model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    # model = torch.hub.load('/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/yolov3', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model


# def load_frcnn(device="cuda:1"):
#     model = FRCNN()
#     return model


def load_frcnn_coco(device="cuda:1"):
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './mmdetection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config, checkpoint, device="cuda:0")
    return model