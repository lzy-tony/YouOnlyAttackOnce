import sys

import torch

sys.path.append("../frcnn")

from frcnn.model import FasterRCNNVGG16
from frcnn.trainer import FasterRCNNTrainer


def load_yolo(model_type='yolov3_spp', device="cuda:1"):
    model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model


def load_frcnn(device="cuda:1"):
    faster_rcnn = FasterRCNNVGG16().to(device)
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    trainer.load("/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/frcnn/chainer_best_model_converted_to_pytorch_0.7053.pth")
    print(next(faster_rcnn.parameters()).device)
    print(next(trainer.parameters()).device)
    return faster_rcnn, trainer