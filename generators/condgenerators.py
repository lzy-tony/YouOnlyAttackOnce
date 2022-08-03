from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchstat as stat

###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

def snlinear(eps=1e-12,device='cuda:0', **kwargs):
    l1 = nn.Linear(**kwargs)
    # l1.weight.requires_grad = False
    l2 = nn.utils.spectral_norm(l1, eps=eps)
    # l2.weight.requires_grad = False
    
    # return l2
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def alf_def(kernel_size=3, pad=2, sigma=1, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gk = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gk = gk / torch.sum(gk)

    # Reshape to 2d depthwise convolutional weight
    gk = gk.view(1, 1, kernel_size, kernel_size)
    gk = gk.repeat(channels, 1, 1, 1)

    gf = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=kernel_size-pad,  bias=False)

    gf.weight.data = gk
    gf.weight.requires_grad = False

    return gf

class ConGeneratorResnet(nn.Module):
    def __init__(self, inception=False, nz=16, layer=1, loc = [1,1,1], data_dim='high',  alf = 'walf_gaussion_s1'):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(ConGeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.alf = alf
        # print('alf', self.alf)
        self.layer = layer
        self.snlinear = snlinear(in_features=80, out_features=nz, bias=False)
        if self.layer > 1:
            self.snlinear2 = snlinear(in_features=nz, out_features=nz, bias=False)
        if self.layer > 2:
            self.snlinear3 = snlinear(in_features=nz, out_features=nz, bias=False)
        self.loc = loc
        # Input_size = 3, n, n
        self.atten_1 = FeatureFusionModule(3+nz * self.loc[0],3+nz * self.loc[0])
        self.atten_2 = FeatureFusionModule(ngf+nz * self.loc[1],ngf+nz * self.loc[1])
        self.atten_3 = FeatureFusionModule(ngf * 2+nz * self.loc[2],ngf * 2+nz * self.loc[2])
        self.reflection = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3+nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False)
        self.block1 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(3+nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf+nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2+nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            # nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        if 'walf_gaussion_s' in self.alf:
            # sigma = int("n")
            # print('sigma:', sigma)
            self.alf_layer = alf_def(kernel_size=55, pad=28)
            

    def forward(self, input, z_one_hot, eps=0.031):
        z_cond = self.snlinear(z_one_hot)
        if self.layer > 1:
            z_cond = self.snlinear2(z_cond)
        if self.layer > 2:
            z_cond = self.snlinear3(z_cond)
        ## loc 0
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        assert self.loc[0] == 1

        x_1 = self.reflection(self.atten_1(input, z_img))
        # x_1 = self.reflection(torch.cat((input, z_img), dim=1))

        x_2 = self.conv1(x_1)
        x = self.block1(x_2)
        # loc 1
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[1]:
            x = self.block2(self.atten_2(x, z_img))
            # x = self.block2(torch.cat((x, z_img), dim=1))
        else:
            x = self.block2(x)
        # loc 2
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[2]:
            x = self.block3(self.atten_3(x, z_img))
            # x = self.block3(torch.cat((x, z_img), dim=1))
        else:
            x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        # scale noise
        x = torch.tanh(x)
        if self.alf:
            x = self.alf_layer(x)
        return x * eps

from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=1, out_chan=2, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        # atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params











class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
    
    
    
if __name__ == '__main__':
    
    netG = ConGeneratorResnet()
    print(netG)
    # net_attention = FeatureFusionModule(5, 6)
    # test_sample = torch.rand(2, 2, 64, 64)
    test_sample_2 = torch.rand(2, 3, 640, 384)
    # a = net_attention(test_sample, test_sample_2)
    label = torch.from_numpy(np.random.choice([35], 2)).long()
    z_class_one_hot = torch.zeros(2, 80).scatter_(1, label.unsqueeze(1), 1)

    # print('Generator output:', netG(test_sample))
    a = netG(test_sample_2, z_class_one_hot)
    print(a.size())
    # print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))
    print(0)
