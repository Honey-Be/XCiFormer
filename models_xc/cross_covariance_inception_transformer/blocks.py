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

from .mixers import *
from patches import LPI
class XCiFormerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                attn=XCMixer, 
                use_layer_scale=False, layer_scale_init_value=1e-5, 
                ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, attention_head=attention_head, pool_size=pool_size,)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, S: tuple[int, int]):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_3 * self.local_mp(self.norm3(x), S[0], S[1]))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.local_mp(self.norm3(x), S[0], S[1]))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x