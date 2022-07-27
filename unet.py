import math
import functools

import torch
import torch.nn as nn


def snlinear(in_features, out_features, bias):
    l1 = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    l2 = nn.utils.spectral_norm(l1, eps=1e-12)
    return l2


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


class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs,
                 output_h, output_w, frames,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        '''
        Unet Generator
        params:
            intput_nc (int) : number of channels in input img
            output_nc (int) : number of channels in output img
            num_downs (int) : number of downsamplings in UNet
                              if |num_downs| = 7, 128 * 128 gets 1 * 1
            ngf(int)        : number of filters in last conv layer
            norm_layer      : normalization layer
        Unet is constructed recursively

        Unet accepts [P, C, H, W] format data, where
            - P is the number of frames from the video
            - C is the number of channels
            - H is height
            - W is width
        Unet:
            - passes through UnetGenerator, obtaining [P, C, H, W]
            - convs to [C, H, W] map
            - adaptive average pool to desired [C, H_out, W_out] map
        '''
        super(Unet, self).__init__()
        self.unetgen = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs, 
                                     ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.alf_layer = alf_def(kernel_size=55, pad=28)
        self.dim_reduction = nn.Conv2d(frames*output_nc, output_nc, 1)
        self.upsample_kernel = nn.AdaptiveAvgPool2d((output_h, output_w))
    
    def forward(self, input):
        unet_out = self.unetgen(input) # P x C x H x W
        gf_out = self.alf_layer(unet_out)
        reshape_out = gf_out.view(-1, gf_out.shape[-2], gf_out.shape[-1])
        dim_reduction_out = self.dim_reduction(reshape_out)
        out = self.upsample_kernel(dim_reduction_out)
        out = torch.tanh(out) / 2 + 0.5 # to [0, 1]
        return out


class UnetGenerator(nn.Module):
    def __init__(self,
                 input_nc, output_nc, num_downs,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model_g = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model_g(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer = nn.BatchNorm2d, use_dropout=False
                 ):
        super(UnetSkipConnectionBlock, self).__init__()
        
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        if outermost:
            # downconv = nn.Conv2d(input_nc+16, inner_nc, kernel_size=4,
            #                     stride=2, padding=1, bias=use_bias)
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                stride=2, padding=1, bias=use_bias)

        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # updrop = nn.Dropout(0.2)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv,upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


if __name__ == '__main__':
    device="cuda"
    netG = Unet(input_nc=4, output_nc=3, num_downs=7, 
                output_h=1260, output_w=2790, frames=30,
                ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False).to(device)
    input = torch.zeros(10, 4, 384, 640).to(device)
    output = netG(input)
    print(output.shape)
