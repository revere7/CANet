#!/usr/bin/env python3

from distutils.errors import DistutilsModuleError
import os
import argparse
import numpy as np
# import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
# import matplotlib.pyplot as plt
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from fixed_filter import all_filter_list
from srm_filter_kernel import all_normalized_hpf_list

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from srm_filter_kernel import all_hpf_list
from MPNCOV.python import MPNCOV
# from non_local_embedded_gaussian import NONLocalBlock2D

IMAGE_SIZE = 256
BATCH_SIZE = 27
DIM = 32
DIM1 = 8
DIM2 = 4
LR = 0.02
WEIGHT_DECAY = 5e-4


TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1

OUTPUT_PATH = Path(__file__).stem


# PARAMS_PATH = os.path.join(OUTPUT_PATH, 'params.pt')
# LOG_PATH = os.path.join(OUTPUT_PATH, 'model_log')

class Quant(nn.Module):
    def __init__(self, quanti):
        super(Quant, self).__init__()

        self.quanti = quanti

    def forward(self, input):
        output = input / self.quanti

        return output


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


def build_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list:
        row_1 = int((5 - hpf_item.shape[0]) / 2)
        row_2 = int((5 - hpf_item.shape[0]) - row_1)
        col_1 = int((5 - hpf_item.shape[1]) / 2)
        col_2 = int((5 - hpf_item.shape[1]) - col_1)
        hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
        filters.append(hpf_item)
    return filters



def build_filters2():
    filters = []
    ksize = [5]     
    lamda = np.pi/2.0 
    sigma = [0.5,1.0]
    phi = [0,np.pi/2]
    for theta in np.arange(0,np.pi,np.pi/8): #gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize[0],ksize[0]),sigma[k],theta,sigma[k]/0.56,0.5,phi[j],ktype=cv2.CV_32F)
                #print(1.5*kern.sum())
                #kern /= 1.5*kern.sum()
                filters.append(kern)
    return filters


class HPF_srm(nn.Module):
    def __init__(self):
        super(HPF_srm, self).__init__()

        filt_list = build_filters()
        # filt_list = np.array([build_filters(),build_filters(),build_filters()])

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        self.tlu = TLU(5.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)

        return output 
    

class HPF_gabor(nn.Module):
    def __init__(self):
        super(HPF_gabor, self).__init__()

        filt_list = build_filters2()
        # filt_list = np.array([build_filters(),build_filters(),build_filters()])

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(32, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        self.tlu = TLU(5.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)

        return output 
    
# CDC convoltion https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/models/CDCNs.py, block1+2 86.10%
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.5):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
        
        
class block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                Conv2d_cd(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                Conv2d_cd(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out


class block2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                Conv2d_cd(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                #nn.AvgPool2d(kernel_size=64, stride=1)
                )

    def forward(self,x):
        out=self.basic(x)
        return out


class block3(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block3, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                #nn.AvgPool2d(kernel_size=32, stride=1)
                )
        self.gap = nn.AvgPool2d(kernel_size=32, stride=1)

    def forward(self,x):
        out = x
        out1 = self.basic(out)

        return out1
    

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # 先LN
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

# The proposed DAFF
class DAFF(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features)
        
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(in_features)
        
        # The reduction ratio is always set to 4
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features//4)
        self.excitation = nn.Linear(in_features//4, in_features)
                
    def forward(self, x):
        #print(x.shape)
        B, N, C = x.size()
        #cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        #x = tokens.reshape(B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C).permute(0, 3, 1, 2)

        output = x.unsqueeze(3)
        x = output.reshape(output.shape[0], output.shape[1], int(output.shape[2]/DIM), int(output.shape[3]*DIM))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = shortcut + x

        x = self.conv3(x)
        output = self.bn3(x)

        #weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        #weight = self.excitation(self.act(self.compress(weight)))
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[3])
        out = output
        #cls_token = cls_token * weight
        
        #tokens = x.flatten(2).permute(0, 2, 1)
        #out = torch.cat((cls_token, tokens), dim=1)
        
        return out


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 论文里的\sqrt{d_k}

        self.attend = nn.Softmax(dim = -1)
        # Q K V一块处理了，有的工作是用了3个维度为dim的Linear层分别得到的
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入 x -> (batch, 197, 768)即(batch, num_patch + 1, hid_dims)
        # 按照最后一维（特征维度)分成3块，分别对应QKV
        # chunk后是一个tuple，即(q, k, v)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v都做维度变换，(batch, 197, 768) -> (batch, 12, 197, 768 / 12 = 64)
        # 12是head的数量，目的是做**多头**注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # Q @ K^T  /  \sqrt{d_k}
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  # softmax

        out = torch.matmul(attn, v)
        # 把多头拼回去 -> (batch, 197, 768)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                #PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            #  先LN再做Att或者Forward
    def forward(self, x):
        for attn in self.layers:  # attn为Multi-head Attention，ff就是FeedForward
            x = attn(x) #, ff + x  # 残差连接，图片中的边线
            #x = ff(x) + x
        return x


class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=32, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention
    

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class MHSA(nn.Module):
    def __init__(self, n_dims, width=32, height=32, heads=1):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1,  bias = True)

    def forward(self, x):
        result = self.depthwise(x)
        return result
    


class LMHSA(nn.Module):
    """
    Lightweight Multi-head-self-attention module.
    Inputs:
        Q: [N, C, H, W]
        K: [N, C, H / stride, W / stride]
        V: [N, C, H / stride, W / stride]
    Outputs:
        X: [N, C, H, W]
    """
    def __init__(self, input_size, channels, d_k, d_v, stride, heads, dropout):
        super(LMHSA, self).__init__()
        self.dwconv_k = DWCONV(channels, channels)
        self.dwconv_v = DWCONV(channels, channels)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2
        self.B = nn.Parameter(torch.Tensor(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)


    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        # x_reshape = nn.LayerNorm(c).cuda()(x_reshape)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        result = result + x
        return result


class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """
    def __init__(self, in_channels, R, dropout = 0.):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.norm = nn.LayerNorm(in_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size = 1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, exp_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(exp_channels, in_channels),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        # x_reshape = nn.LayerNorm(c).cuda()(x_reshape)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))
  
        #x1 = x_reshape.view(b, c, h, w)
        x1 = x_reshape
        out = self.mlp(x1)
        

        output = out.unsqueeze(3)
        output = output.reshape(output.shape[0], int(output.shape[1]/DIM2), int(output.shape[2]/DIM1), int(output.shape[3]*DIM))

        #print(output.shape)

        #x1 = self.norm(x)
        result = x + output
        return result


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """
    def __init__(self, in_channels, out_channels):
        super(LPU, self).__init__()
        self.DWConv = DWCONV(in_channels, out_channels)

    def forward(self, x):
        result = self.DWConv(x) + x
        return result
    

class CMTBlock(nn.Module):
    def __init__(self, img_size, stride, d_k, d_v, num_heads, R = 3.6, in_channels = 46):
        super(CMTBlock, self).__init__()

        # Local Perception Unit
        #self.lpu = LPU(in_channels, in_channels)

        # Lightweight MHSA
        self.lmhsa = LMHSA(img_size, in_channels, d_k, d_v, stride, num_heads, 0.0)

        # Inverted Residual FFN
        self.irffn = IRFFN(in_channels, R)

    def forward(self, x):
        #x = self.lpu(x)
        x = self.lmhsa(x)
        #x = self.irffn(x)
        return x




class NoiseEnhance(nn.Module):
    def __init__(self, img):
        super(NoiseEnhance, self).__init__()

        self.media_list = cv2.medianBlur(img, 3)
        self.sig = nn.Sigmoid()

        self.basic=nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

    def forward(self, input):
        out = input
        output = self.media_list(input)
        output = self.sig(out - output)
        output = self.basic(output)

        return output 


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    

class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        # self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        # x_srm = self.srm(x)
        fea = self.conv(x_srm)        
        att_map = self.pa(fea)
        
        return att_map



class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.relu = nn.ReLU()

    self.group1 = HPF_srm()
    self.group2 = block1(186,64)
    self.group3 = block1(64,128)
    self.group4 = block2(128,256)
    self.group5 = block3(128,256)

    self.group11 = HPF_gabor()
    
    self.srm_sa = SRMPixelAttention(96)
    self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
    )

    #self.median = NoiseEnhance()
    self.cmt_layer = CMTBlock(img_size = 32, stride = 1, d_k = 256, d_v = 256, num_heads = 4, R = 4, in_channels = 256)

    self.transformer = Transformer(dim=1024, depth=1, heads=2, dim_head=4, mlp_dim=1024*4) 
    self.mhsa = MHSA(n_dims=1024)
    self.chan = SE(64,16)
    self.channelatt = ChannelAttentionModule()
    self.dual_cma0 = DualCrossModalAttention(in_dim=256, ret_att=False)
    self.dual_cma1 = DualCrossModalAttention(in_dim=256, ret_att=False)

    self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    self.gap = nn.AvgPool2d(kernel_size=16, stride=1)
    self.fc0 = nn.Linear(256, 2)

    self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2048)
    self.dropout = nn.Dropout(p=0.7)
    self.fc2 = nn.Linear(2048, 2)

  def forward(self, input):
    output = input
    
    output_y = output[:, 0, :, :]
    output_u = output[:, 1, :, :] 
    output_v = output[:, 2, :, :] 
    out_y = output_y.unsqueeze(1)
    out_u = output_u.unsqueeze(1)
    out_v = output_v.unsqueeze(1)
    y = self.group1(out_y)
    u = self.group1(out_u)
    v = self.group1(out_v)
    output = torch.cat([y, u, v], dim=1)

    yy = self.group11(out_y)
    uu = self.group11(out_u)
    vv = self.group11(out_v)
    output11 = torch.cat([yy, uu, vv], dim=1)

    output = torch.cat([output,output11],dim=1)

    output = self.group2(output)

    self.att_map = self.srm_sa(output11)
    x = output * self.att_map + output
    output = self.srm_sa_post(x)

    output = self.group3(output)
    output = self.group4(output)

    #output = self.group5(output)
    #output1 = self.cmt_layer(output)
    output1 = self.channelatt(output)
    output = output1 + output
    #output1 = self.group5(outputt)


    #output1 = output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[3])
    #print(output.shape) # [16,256,1024]
    #output1 = self.transformer(output1)  # transformer block
    #print(output.shape) # [16,256,1024]
    #output1 = output1.unsqueeze(3)
    #print(output.shape)
    #output1 = output1.reshape(output1.shape[0], output1.shape[1], int(output1.shape[2]/DIM), int(output1.shape[3]*DIM))
    #print(output.shape)
    #output = self.att(output)
    #output = output1 + output

    output = MPNCOV.CovpoolLayer(output)
    output = MPNCOV.SqrtmLayer(output, 5)
    output = MPNCOV.TriuvecLayer(output)
    #output = self.gap(output)

    output = output.view(output.size(0), -1)
    #output = self.fc0(output)
    output = self.fc1(output)
    output = self.dropout(output)
    output = self.fc2(output)

    return output


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()  # ONE EPOCH TRAIN TIME
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)
        # data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
      }
    torch.save(all_state, PARAMS_PATH1)

    if accuracy > best_acc and epoch > TMP:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        #nn.init.constant_(module.bias.data, val=0)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[2, 3]).copy()  # gaiwei [2,3]

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # data = np.expand_dims(data, axis=1) ##delete
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform

        self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.ppm'
        self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.ppm'

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        cover_path = self.bossbase_cover_path.format(file_index)
        stego_path = self.bossbase_stego_path.format(file_index)

        cover_data = cv2.imread(cover_path, -1)
        cover_data = cv2.cvtColor(cover_data, cv2.COLOR_BGR2RGB)
        cover_data = np.transpose(cover_data, (2, 0, 1))
        stego_data = cv2.imread(stego_path, -1)
        stego_data = cv2.cvtColor(stego_data, cv2.COLOR_BGR2RGB)
        stego_data = np.transpose(stego_data, (2, 0, 1))

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):
    #  setLogger(LOG_PATH, mode='w')

    #  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    # mp.set_start_method('spawn')
    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE
    TIMES = args.times


    BOSSBASE_COVER_DIR = '/home/weikangkang/data/ALASKA20000_Spatial1/ALASKA_v2_TIFF_256_COLOR_RANDOM2'
    BOSSBASE_STEGO_DIR = '/home/weikangkang/data/ALASKA20000_Spatial1/ALASKA_v2_TIFF_256_COLOR_HILL-GINA_0.2'

    TRAIN_INDEX_PATH = 'index_list/alaska_train_index_14000.npy'
    VALID_INDEX_PATH = 'index_list/alaska_val_index_1000.npy'
    TEST_INDEX_PATH = 'index_list/alaska_test_index_5000.npy'

    LOAD_RATE = float(EMBEDDING_RATE) + 0.1
    LOAD_RATE = round(LOAD_RATE, 1)

    PARAMS_NAME = '{}-{}-{}-params-{}-lr-t5={}-saca-trans.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    PARAMS_NAME1 = '{}-{}-{}-process-params-t5-{}-lr={}-saca-trans.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    LOG_NAME = '{}-{}-{}-model_log-{}-lr-t5={}-saca-trans.log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, LR)
    #statePath='./Test5/'+PARAMS_NAME1

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    # transfer learning
    PARAMS_INIT_NAME = 'HILL-GINA-0.3-1-params--lr-t5=0.02-saca-trans.pt'#.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES,LR)

    #if LOAD_RATE == 0.4:
        #PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.01')
        
    #if LOAD_RATE == 0.3:
        #PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.01')


    PARAMS_INIT_PATH = os.path.join(OUTPUT_PATH, PARAMS_INIT_NAME)

    setLogger(LOG_PATH, mode='w')

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = Net().to(device)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

    # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)
    #EPOCHS = 230
    #DECAY_EPOCH = [80, 130, 170]
    #TMP = 170
    
    EPOCHS = 120
    DECAY_EPOCH = [50, 80, 100]
    TMP = 100

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1

    if LOAD_RATE != 0.5:
        all_state = torch.load(PARAMS_INIT_PATH)
        original_state = all_state['original_state']
        model.load_state_dict(original_state)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc = 0.0
    for epoch in range(startEpoch, EPOCHS + 1):
        scheduler.step()

        train(model, device, train_loader, optimizer, epoch)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)

    logging.info('\nTest set accuracy: \n')

    # load best parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)


def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--DATASET_INDEX',
        help='Path for loading dataset',
        type=str,
        default='1'
    )

    parser.add_argument(
        '-alg',
        '--STEGANOGRAPHY',
        help='embedding_algorithm',
        type=str,
        choices=['HILL-CMDC', 'SUNIWARD-CMDC'],
        # required=True
        default='HILL-GINA'
    )

    parser.add_argument(
        '-rate',
        '--EMBEDDING_RATE',
        help='embedding_rate',
        type=str,
        choices=[ '0.2', '0.3', '0.4'],
        # required=True
        default='0.2'
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3'],
        #required=True
        default='2,3'
    )

    parser.add_argument(
        '-t',
        '--times',
        help='Determine which gpu to use',
        type=str,
        # required=True
        default=''
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = myParseArgs()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)


