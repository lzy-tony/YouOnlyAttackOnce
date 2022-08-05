import torch    
import cv2
import numpy as np
import torchvision.transforms as transforms
# from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

from util.tensor2img import tensor2img
from cam.eigen_cam import EigenCAM


image_path = "/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/datasets/image/2_220.jpg"
img = np.array(Image.open(image_path))
img = cv2.resize(img, (640, 384))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov5_master', 'yolov5l', source='local')
model.cpu()
print(model.model.model.model[-2])
target_layers = [model.model.model.model[-2]]

# model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', 'yolov3_spp', source='local')
# model.cpu()
# print(model.model.model[-2])
# target_layers = [model.model.model[-2]]


'''
cam = EigenCAM(model, target_layers, use_cuda=False)
out = cam(tensor)
print(out.shape)
# grayscale_cam = cam(tensor)[0, :, :]
# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
# Image.fromarray(cam_image).save("./test_ori.png")
'''

print(tensor.shape)
cam = EigenCAM(model, target_layers, use_cuda=False)
out = cam(tensor)
print(out.shape)
grayscale_cam = out[0, :, :].cpu().numpy()
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
print(cam_image.shape)
Image.fromarray(cam_image).save("./test_tensor.png")

img = torch.zeros(3, 384, 640)
img[0] = out[0, :, :]
img[1] = out[0, :, :]
img[2] = out[0, :, :]
tensor2img(img, "test_mask.png")