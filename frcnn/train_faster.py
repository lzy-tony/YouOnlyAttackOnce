from readline import remove_history_item
import torch
import numpy as np
import json
import os
import math
import time
import torchvision
import cv2
from PIL import Image
from copy import copy
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from frcnn import FRCNN

unloader = transforms.ToPILImage()
frcnn = FRCNN()
crop = False
count = False

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor.shape)
                print(' - gradient:', tensor.grad.shape)
                print()
            except AttributeError as e:
                getBack(n[0])

torch.autograd.set_detect_anomaly(True)
num = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
input_path = "/home/duanchengqi20/Patch/image/1_223.jpg"
# model = torch.hub.load('/home/duanchengqi20/.cache/torch/hub/ultralytics_yolov3_master', 'yolov3', source="local").to(device)  # or yolov3-spp, yolov3-tiny, custom
# model.eval()


with open("/home/duanchengqi20/Patch/loc.json", "r") as f:
    loc = json.load(f)

train_img = []


class mydataset(data.Dataset):
    def __init__(self, data):
        self.x = data
        self.len = len(data)

    def __getitem__(self, item):
        return self.x[item].cnt

    def __len__(self):
        return self.len


def patch_init():
    a = np.random.randint(255, size=(1080, 1920, 3))
    a = torch.autograd.Variable(torch.from_numpy(a).type(torch.float32), True)
    return a.to(device)

cnt = 0 


class img():
    def __init__(self, path, name, cnt):
        self.image = Image.open(path)
        to_tensor = transforms.ToTensor()
        # pic = Image.open(path)
        # self.tensor = img_to_tensor(pic)
        # self.tensor = self.tensor.transpose(0,1).transpose(1,2).to(device)
        self.tensor = to_tensor(self.image).type(torch.float32).to(device)
        self.tensor *= 255
        self.tensor = self.tensor.transpose(0,1).transpose(1,2)
        self.name = name
        self.loc = loc[self.name]
        self.shape = self.tensor.shape
        self.cnt = cnt

    def init_patch(self):
        bg = torch.zeros(self.shape).to(torch.float32)
        a = np.random.randint(255, size=(self.loc[3], self.loc[2], 3))
        a = torch.from_numpy(a).type(torch.float32)
        bg[self.loc[1]:self.loc[1]+self.loc[3] , self.loc[0]:self.loc[0]+self.loc[2] , :] = a
        return bg

    def transform_patch(self, patch):
        patch = patch.transpose(2,1).transpose(0,1)
        img_reshaper = torch.nn.AdaptiveAvgPool2d((self.loc[3], self.loc[2])).to(device)
        reshaped_patch = img_reshaper(patch)
        reshaped_patch1 = F.pad(reshaped_patch,[self.loc[0],self.shape[1]-self.loc[0]-self.loc[2],self.loc[1],self.shape[0]-self.loc[1]-self.loc[3]])
        reshaped_patch2 = reshaped_patch1.transpose(0,1).transpose(1,2)
        # getBack(reshaped_patch.grad_fn)
        return reshaped_patch2

    def add_patch(self, patch):
        patch = patch.to(device)
        mask = torch.ones(self.shape)
        mask[self.loc[1]:self.loc[1]+self.loc[3] , self.loc[0]:self.loc[0]+self.loc[2] , :] = 0
        mask = mask.to(device)
        return self.tensor * mask + patch * (1 - mask)


    def save_img(self, adv, name):
        adv = adv.squeeze(0)
        if adv.shape[0] != 3:
            adv = adv.transpose(2,1).transpose(1,0)
        adv = (adv + 0.5).detach().cpu().numpy().transpose(1,2,0).astype("uint8")
        # toPIL = transforms.ToPILImage()
        # pic = toPIL(adv)
        # pic.save('/home/duanchengqi20/Patch/image/trained/test{}.png'.format(name))
        Image.fromarray(adv).save('/home/duanchengqi20/Patch/image/trained/test{}.png'.format(name))


    def attack(self, patch):
        global cnt
        # mask = torch.ones(self.shape)
        # mask[self.loc[1]:self.loc[1]+self.loc[3] , self.loc[0]:self.loc[0]+self.loc[2] , :] = 0
        # mask = mask.to(device)
        # adv_x = (self.tensor * mask + patch * (1 - mask)).transpose(1,2).transpose(0,1).unsqueeze(0).to(torch.float32) / 255
        adv_x = self.add_patch(patch).transpose(2,1).transpose(1,0).unsqueeze(0).to(torch.float32)
        # adv_x = self.add_patch(patch).transpose(1,2).transpose(0,1).unsqueeze(0).to(torch.float32) / 255
        #self.save_img(adv_x, cnt)
        cnt += 1
        return adv_x


def los(_bboxes, _labels, _scores):
    l = 0
    for i, labels in enumerate(_labels):
        # vis_bbox(at.tonumpy(files[i].squeeze(0)),
        #     at.tonumpy(_bboxes[i]),
        #     at.tonumpy(_labels[i]).reshape(-1),
        #     at.tonumpy(_scores[i]).reshape(-1))
        for j, label in enumerate(labels):
            if label == 5 or label == 6:
                l += _scores[i][j]
    
    return l
t = 0
def val():
    image = Image.open("/home/duanchengqi20/Patch/image/trained/test99545.png")
    to_tensor = transforms.ToTensor()
    print((to_tensor(image).unsqueeze(0).cuda()-t).abs().max())
    a, b, c = frcnn.detect_image([to_tensor(image).unsqueeze(0).cuda()], crop = crop, count = count, pil=False)
    # r_image = frcnn.detect_image(image, crop = crop, count = count, pil = True)
    print(1)

input_path = "/home/duanchengqi20/Patch/image/raw"
files = os.listdir(input_path)
for i, file in enumerate(files):
    # if i>32:
    if "4_229" not in file:
        continue
    #     continue
    if "231" not in file and "230" not in file and "229" not in file and "228" not in file and "227" not in file and "226" not in file:
        continue
    if file not in loc.keys():
        continue
    # if i > 0:
    #     break
    train_img.append(img(os.path.join(input_path, file), file, num))
    num += 1
    # if num > 8:
    #     break

train_set = mydataset(train_img)
origin_patch = patch_init()
origin_patch.retain_grad()
batch_size = 1

train_data_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last = True)
num_epoch = 5000
reshaped_patch = []
adv = []

for epoch in range(num_epoch):
    total = 0
    print("EPOCH:{} HAS STARTED".format(epoch))
    for x in train_data_loader:
        a = time.time()
        x = [train_img[i] for i in x]
        reshaped_patch.clear()
        adv.clear()
        reshaped_patch = [i.transform_patch(origin_patch) for i in x]
        adv = [i.attack(reshaped_patch[t])/255 for t, i in enumerate(x)]
        #img_reshaper = torch.nn.AdaptiveAvgPool2d((384, 640)).to(device)
        # loss_list1 = [img_reshaper(i) for i in loss_list]
        # aaa = torch.concat(loss_list, 0)
        #[train_img[0].tensor.unsqueeze(0).transpose(3,2).transpose(2,1)]
        #frcnn.net = frcnn.net.train()
        label, confidence, bboxes = frcnn.detect_image(adv, crop = crop, count = count, pil = False)
        #label1, confidence1, bboxes1 = frcnn.detect_image(Image.open("/home/duanchengqi20/Patch/image/raw/4_229.jpg"), crop = crop, count = count, pil = True)
        #frcnn.net = frcnn.net.eval()
        # print("1:{}".format(torch.cuda.memory_allocated(0)))   
        total_loss = los(bboxes, label, confidence)
        print(total_loss)
        if total_loss == 0:
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            continue
        total += total_loss.data
        total_loss = total_loss / batch_size
        try:
            total_loss.backward()
        except:
            print("Anomaly")
            total_loss = None
            print("1:{}".format(torch.cuda.memory_allocated(0)))

        # getBack(loss_list[0][1].grad_fn)
        #print(origin_patch.grad)
        origin_patch -= origin_patch.grad * (1e4 + 10*(1-epoch/1.5*num_epoch)*0.5/float(abs(origin_patch.grad).max()))
        #origin_patch -= origin_patch.grad*1e5
        origin_patch.clamp(0, 255)
    print(total)
    train_img[0].save_img(origin_patch, epoch)
    
    if total/8 <= 0.01 or epoch == num_epoch - 1:
        # train_img[0].save_img(origin_patch, 100909)
        for i,j in enumerate(train_img):
            reshaped_pat = j.transform_patch(origin_patch)
            adv_x = j.attack(reshaped_pat)
            train_img[0].save_img(adv_x, 99545)
            label, confidence, bboxes = frcnn.detect_image([adv_x/255], crop = crop, count = count, pil = False)
            t = adv_x/255
        val()
        exit(0)






