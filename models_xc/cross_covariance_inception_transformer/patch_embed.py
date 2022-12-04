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
from mixers import *

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, kernel_size=16,  stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0,2,3,1)
        return x

class FirstPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, kernel_size=3,  stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()
        
        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.gelu1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm2 = nn.BatchNorm2d(embed_dim)
        
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)    
        x = x.permute(0,2,3,1)
        return x