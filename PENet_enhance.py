import math
import cv2
import numpy as np
import torch
import torch.nn as nn
# from ...builder import BACKBONES
# from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
#                       kaiming_init)
# from mmcv.runner import BaseModule
import torch.nn.functional as F



# device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)#5x5的高斯滤波器

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        #torch.nn.functional.pad()函数可以将一个Tensor类型的变量在不改变维度的情况下扩展到固定长度。
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):#高斯下采样
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x,current):#高斯上采样
        up = torch.zeros((x.size(0), x.size(1), current.size(2), current.size(3)),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        # print(img.device)#cpu
        self.kernel = self.kernel.to(img.device)#5x5的高斯滤波器
        current = img#torch.Size([1, 3, 640, 640])
        pyr = []
        for _ in range(self.num_high):
            #每次高斯金字塔操作后，图像的宽度和高度减半，这意味着分辨率为原来的1/4
            down = self.pyramid_down(current)#论文中的G(x)=Down(Gaussian(x))
            # print(down.shape)#torch.Size([1, 3, 218, 426])
            # print('down:',down.shape)#torch.Size([1, 3, 320, 320])->torch.Size([1, 3, 160, 160])->torch.Size([1, 3, 80, 80])
            #为了在向上采样时能够恢复具有较高分辨率的原始图像，就要获取在采样过程中所丢失的信息，这些丢失的信息就构成了拉普拉斯金字塔
            up = self.upsample(down,current)
           
            # print('up:',up.shape)#torch.Size([1, 3, 640, 640])->torch.Size([1, 3, 320, 320])->torch.Size([1, 3, 160, 160])
            
            diff = current - up#拉普拉斯金字塔
            # print('diff:',diff.shape)#torch.Size([1, 3, 640, 640])->torch.Size([1, 3, 320, 320])->torch.Size([1, 3, 160, 160])
            pyr.append(diff)
            current = down
        
        pyr.append(current)#torch.Size([1, 3, 80, 80])
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image,level)#上采样两倍
            image = up + level
        return image


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2d(in_features, out_features, 3, padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))


# @BACKBONES.register_module()
class PENet(nn.Module):

    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high#分为4层
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)

        for i in range(0, self.num_high + 1):
            self.__setattr__('AE_{}'.format(i), AE(3))

    def forward(self, x):
        #拉普拉斯分解为4个分辨率：torch.Size([1, 3, 640, 640])、torch.Size([1, 3, 320, 320])、torch.Size([1, 3, 160, 160])、torch.Size([1, 3, 80, 80])
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []

        #对分量进行处理（该部分直接看AE模块),从最低分辨率到高分辨率的顺序进入AE网络中增强
        for i in range(self.num_high + 1):
            trans_pyr = self.__getattr__('AE_{}'.format(i))(
                pyrs[-1 - i])
            trans_pyrs.append(trans_pyr)
        #trans_pyrs中四个分辨率特征为：torch.Size([1, 3, 80, 80]) torch.Size([1, 3, 160, 160]) torch.Size([1, 3, 320, 320]) torch.Size([1, 3, 640, 640])
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)#合并,torch.Size([1, 3, 640, 640])
        return out#增强后的图片


class DPM(nn.Module):#用于捕获远程依赖关系
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), bias=False):
        super(DPM, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)#torch.Size([1, 32, 1, 1])

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)#torch.Size([1, 32, 1, 1])
        x = x + channel_add_term#torch.Size([1, 32, 80, 80])
        return x


import cv2
from torchvision import transforms


def sobel(img):
    add_x_total = torch.zeros(img.shape)

    for i in range(img.shape[0]):
        # x = img[i, :, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)#(80, 80, 3)->(160, 160, 3)->(320,320, 3)->(640, 640, 3)
        x = img[i, :, :, :].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)#(80, 80, 3)->(160, 160, 3)->(320,320, 3)->(640, 640, 3)
        x = x * 255
        #为了避免信息丢失，在计算时要先使用更高的数据类型 cv2.CV_64F，再通过取绝对值将其映射为 cv2.CV_8U（8 位图）类型。
        #所以，通常要将函数 cv2.Sobel()内参数 ddepth 的值设置为“cv2.CV_64F”
        x_x = cv2.Sobel(x, cv2.CV_64F, 1, 0)#计算水平方向的边缘
        x_y = cv2.Sobel(x, cv2.CV_64F, 0, 1)##计算垂直方向的边缘
        add_x = cv2.addWeighted(x_x, 0.5, x_y, 0.5, 0)#在水平、垂直两个方向叠加的边缘信息
        add_x = transforms.ToTensor()(add_x).unsqueeze(0)
        add_x_total[i, :, :, :] = add_x

    return add_x_total

#四个不同的分辨率都要经过AE网络进行增强
class AE(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), groups=1):
        super(AE, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction
        self.agg = nn.Conv2d(6,
                             3,
                             1,
                             stride=1,
                             padding=0,
                             bias=False)#用于调整通道数输出
        self.conv_edge = nn.Conv2d(3, 3, kernel_size=1, bias=bias)

        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 3)
        self.dpm = nn.Sequential(DPM(32, 32))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1)
        self.lpm = LowPassModule(32)#动态低低通道滤波器
        self.fusion = nn.Conv2d(6, 3, kernel_size=1)

    def forward(self, x):
        #x:torch.Size([1, 3, 80, 80]) 

        #细节处理模块(DPM)中的EB块 边缘分支：有助于模型更好地识别目标的轮廓和边缘特征，并增强目标组件的纹理信息
        s_x = sobel(x)#torch.Size([1, 3, 80, 80]) 
        s_x=s_x.to(x.device)#自己加的
        s_x = self.conv_edge(s_x)#torch.Size([1, 3, 80, 80])
      
        #细节处理模块(DPM)中的CB块  上下文分支：获取上下文信息，通过捕捉远程依赖关系来理解目标周围的环境，提高目标检测的准确性
        res = self.res1(x)#第一个残差块,torch.Size([1, 32, 80, 80])
        res = self.dpm(res)#torch.Size([1, 32, 80, 80])
        res = self.res2(res)#第二个残差块 torch.Size([1, 3, 80, 80])
        out = torch.cat([res, s_x + x], dim=1)#torch.Size([1, 6, 80, 80])
        out = self.agg(out)#torch.Size([1, 3, 80, 80])

        #低频增强滤波器(LEF)
        #在每个尺度分量中，低频分量具有图像中大部分的语义信息，它们是检测器预测的关键信息
        low_fea = self.conv1(x)#1x1卷积,torch.Size([1, 32, 80, 80])
        low_fea = self.lpm(low_fea)#动态低低通道滤波器,torch.Size([1, 32, 80, 80])
        low_fea = self.conv2(low_fea)#torch.Size([1, 3, 80, 80])

        #将DPM和LEF的结果进行拼接
        out = torch.cat([out, low_fea], dim=1)#torch.Size([1, 6, 80, 80])
        out = self.fusion(out)#torch.Size([1, 3, 80, 80])
        return out


#使用动态低低通道滤波器来捕获低频信息
#考虑到Inception的多尺度结构，采用大小为1×1,2×2、3×3、6×6的自适应平均池，并在每个尺度末端采用上采样来恢复特征的原始大小
class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        #feats:torch.Size([1, 32, 80, 80])
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)#通道分离为四个部分
        #四个平均池化后的shape：torch.Size([1, 8, 1, 1])、torch.Size([1, 8, 2, 2])、torch.Size([1, 8, 3, 3])、torch.Size([1, 8, 6, 6])
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        #四个上采样后的shape都是torch.Size([1, 8, 80, 80]) 

        bottle = torch.cat(priors, 1)#torch.Size([1, 32, 80, 80])
        return self.relu(bottle)
