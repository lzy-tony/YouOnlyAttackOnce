import torch


def load_yolo(model_type='yolov3_spp', device="cuda:1"):
    model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model
