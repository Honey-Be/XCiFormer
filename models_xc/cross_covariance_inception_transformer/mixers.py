import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import warnings
from timm.models.layers.helpers import to_2tuple

def floored_sqrt(n: int) -> int:
    if n > 3:
        return max(filter(lambda m: (m*m) <= n, range(0, n)))
    elif n > 0:
        return 1
    elif n < 0:
        raise ArithmeticError
    else:
        return 0

class XCHighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()
        
        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2
        
        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()
        
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W
        
        cx = x[:,:self.cnn_in,:,:].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)
        
        px = x[:,self.cnn_in:,:,:].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)
        
        hx = torch.cat((cx, px), dim=1)
        return hx

class XCLowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
        **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()
        

    def att_fun(self, q, k, v, B, N, C):
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = (attn @ v).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N**0.5), int(N**0.5))#.permute(0, 3, 1, 2)
        
        xa = self.uppool(xa)
        return xa

class XCMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2, 
        **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        
        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim
        
        
        self.high_mixer = XCHighMixer(high_dim)
        self.low_mixer = XCLowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size,)

        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        
        hx = x[:,:self.high_dim,:,:].contiguous()
        hx = self.high_mixer(hx)
        
        lx = x[:,self.high_dim:,:,:].contiguous()
        lx = self.low_mixer(lx)
            
        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x