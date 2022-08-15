import torchvision


class Fcos():
    def __init__(self):
        self.model = torchvision.models.detection.fcos_resnet50_fpn(weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
        pass

