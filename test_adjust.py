import torchvision
import torch
import numpy as np
from PIL import Image

patch = torch.from_numpy(np.array(Image.open("./pgd_smooth_5e-5_epoch4.png")).astype(np.float32))
patch = patch.transpose(1,2).transpose(0,1)/255
patch = torchvision.transforms.functional.adjust_brightness(patch,114514)
patch = patch.transpose(0,1).transpose(1,2) * 255
print(patch.max())
Image.fromarray(patch.detach().cpu().numpy().astype("uint8")).save("./saved_patch.png")