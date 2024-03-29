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

sys.path.append("./frcnn")


from util.load_detector import load_frcnn, load_yolo
from util.dataloader import ImageLoader
from util.loss import Faster_RCNN_loss, Original_loss_gpu
from util.tensor2img import tensor2img

sys.path.append("target_models/DINO")
from target_models.DINO.run_dino import MyDino


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default="7e-3", help="size of gradient update")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs to attack")
    parser.add_argument("--batch-size", type=int, default=12, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:2", help="device")
    parser.add_argument("--momentum_beta", type=float, default=0.9, help="momentum need an beta arg")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.device)
    return opt


def train(opt):
    device = opt.device
    beta = opt.momentum_beta
    dataset = ImageLoader()
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    # frcnn = load_frcnn(device=device)
    # frcnn_loss = Faster_RCNN_loss()

    # yolo = load_yolo(device=device)
    # yolo_loss = Original_loss_gpu(yolo)
    
    dino = MyDino()

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
    mom_grad = torch.zeros((3, patch_height, patch_width)).to(device)
    mask = torch.ones((3, patch_height, patch_width)).to(device)

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")

        for batch, (img, pos, name) in enumerate(tqdm(dataloader)):
            noise.requires_grad = True

            tyt, txt, twt, tht = pos
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
                
                    
                small_noise = transform_kernel(noise)
                small_mask = transform_kernel(mask)
                ori = im[..., ux:dx, uy:dy]
                ori = ori.unsqueeze(dim=0)
                patch = small_noise * small_mask + ori * (1 - small_mask)
                pad_patch = F.pad(patch, p2d, "constant", 0)

                adv_im = im * (1 - im_mask) + im_mask * pad_patch                
                output_dino = dino(adv_im)
                loss3 = dino.cal_loss(output_dino)
                grad3_ = torch.autograd.grad(loss3, noise, retain_graph=False, create_graph=False)[0]
                if not torch.isnan(grad3_[0,0,0]):
                    grad += grad3_
                
                if batch % 10 == 0:
                    tensor2img(adv_im, f"./saves/adv_im_{batch}_{i}.png")
            
            mom_grad = beta * mom_grad + (1-beta) * grad.sign()
            noise = noise.detach() - opt.alpha * mom_grad
            noise = torch.clamp(noise, min=0, max=1)

        
        tensor2img(noise, f"./submission/pgd_dino/pgd_ensemble2_epoch{epoch}.png")
        tensor2img(mask, f"./submission/pgd_dino/mask.png")


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
