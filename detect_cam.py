import torch    
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

COLORS = np.random.uniform(0, 255, size=(80, 3))


image_path = "/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/datasets/image/3_201.jpg"
img = np.array(Image.open(image_path))
img = cv2.resize(img, (640, 384))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov5_master', 'yolov5l', source='local')
model.cpu()
target_layers = [model.model.model.model[-2]]

# cam = EigenCAM(model, target_layers, use_cuda=False)
cam = AblationCAM(model, target_layers, use_cuda=False)
grayscale_cam = cam(tensor)[0, :, :]
print(grayscale_cam.shape)
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image).save("./test_ori.png")

