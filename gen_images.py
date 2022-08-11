import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from util.dataloader import ImageLoader
from util.tensor2img import tensor2img


def gen_images(patch_path, save_path, device, dataset):
    # patch size
    patch_height = 1260
    patch_width = 2790
    # seed size
    seed_height = 1280
    seed_width = 2560
    # yolo input size
    im_height = 384
    im_width = 640
    # video image size
    read_height = 1080
    read_width = 1920
    # calc pad offsets
    r = min(im_height / read_height, im_width / read_width)
    r = min(r, 1.0)
    new_read_height, new_read_width = int(round(read_height * r)), int(round(read_width * r))
    dh, dw = im_height - new_read_height, im_width - new_read_width
    dh /= 2
    dw /= 2

    # read patch
    noise = cv2.imread(patch_path)
    noise = noise.transpose((2, 0, 1))[::-1]
    noise = np.ascontiguousarray(noise)
    noise = torch.from_numpy(noise).to(device)
    noise = noise.float()
    noise /= 255

    for im, label, name in tqdm(dataset):
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        
        
        ty, tx, tw, th = label
        ux = int(round(dh + tx * r))
        uy = int(round(dw + ty * r))
        dx = int(round(dh + (tx + th) * r))
        dy = int(round(dw + (ty + tw) * r))

        transform_kernel = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
        im_mask = torch.ones((dx - ux, dy - uy)).to(device)
        patch = transform_kernel(noise)

        p2d = (uy, im_width - dy, ux, im_height - dx)
        pad_patch = F.pad(patch, p2d, "constant", 0)
        im_mask = F.pad(im_mask, p2d, "constant", 0)

        adv_im = im * (1 - im_mask) + im_mask * pad_patch
        tensor2img(adv_im, save_path + "/" + name[:-3] + "png")        


if __name__ == '__main__':
    p_path = "./submission/Unet5/psgan2_epoch50.png"
    # p_path = "./submission/pgd/texture.png"
    save_path = "./gen_results"
    dataset = ImageLoader()
    device = "cuda:1"
    gen_images(p_path, save_path, device, dataset)