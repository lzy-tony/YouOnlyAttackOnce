import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from util.load_detector import load_yolo
from util.dataloader import ImageLoader
from util.loss import Original_loss_gpu
from util.tensor2img import tensor2img


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default="5e-2", help="size of gradient update")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs to attack")
    parser.add_argument("--batch-size", type=int, default=12, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.device)
    return opt


def train(opt):
    device = opt.device
    dataset = ImageLoader()
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    yolo = load_yolo(device=device)
    compute_loss = Original_loss_gpu(yolo)

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

    noise = torch.zeros((3, patch_height, patch_width)).to(device)
    # mask = torch.ones((3, patch_height, patch_width)).to(device)
    mask = torch.ones((3, int(patch_height / 2), int(patch_width / 2))).to(device)
    pmask = (int(np.ceil(patch_width / 4)), int(np.floor(patch_width / 4)), int(np.ceil(patch_height / 4)), int(np.ceil(patch_height / 4)))
    mask = F.pad(mask, pmask, "constant", 0)

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")

        for batch, (img, label, name) in enumerate(tqdm(dataloader)):
            noise.requires_grad = True

            tyt, txt, twt, tht = label
            img = img.to(device)

            grad = torch.zeros_like(noise, device=device)
            
            for i in range(img.shape[0]):
                im = img[i]
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0

                ty, tx, tw, th = tyt[i].item(), txt[i].item(), twt[i].item(), tht[i].item()
                ux = int(round(dh + tx * r))
                uy = int(round(dw + ty * r))
                dx = int(round(dh + (tx + th) * r))
                dy = int(round(dw + (ty + tw) * r))

                transform_kernel = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
                im_mask = torch.ones((dx - ux, dy - uy)).to(device)
                small_noise = transform_kernel(noise)
                small_mask = transform_kernel(mask)
                ori = im[..., ux:dx, uy:dy]
                ori = ori.unsqueeze(dim=0)
                patch = small_noise * small_mask + ori * (1 - small_mask)

                p2d = (uy, im_width - dy, ux, im_height - dx)
                pad_patch = F.pad(patch, p2d, "constant", 0)
                im_mask = F.pad(im_mask, p2d, "constant", 0)

                adv_im = im * (1 - im_mask) + im_mask * pad_patch
                
                pred = yolo(adv_im)
                loss, _ = compute_loss(pred)

                grad_ = torch.autograd.grad(loss, noise,
                                            retain_graph=False, create_graph=False)[0]
                if not torch.isnan(grad_[0, 0, 0]):
                    grad += grad_
                
                if batch % 10 == 0:
                    tensor2img(adv_im, f"./saves/adv_im2_{batch}_{i}.png")
            
            noise = noise.detach() - opt.alpha * grad.sign()
            noise = torch.clamp(noise, min=0, max=1)

        
        tensor2img(noise, f"./submission/pgd_small/pgd_small_epoch{epoch}.png")
        tensor2img(mask, f"./submission/pgd_small/mask.png")


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
