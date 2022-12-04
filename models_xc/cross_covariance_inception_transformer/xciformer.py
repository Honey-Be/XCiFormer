# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inception transformer implementation.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
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


_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# default_cfgs = {
#     'xciformer_224': _cfg(),
#     'xciformer_384': _cfg(input_size=(3, 384, 384), crop_pct=1.0),
# }


default_cfgs = {
    'xciformer_small': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_small.pth'),
    'xciformer_base': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_base.pth'),
    'xciformer_large': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_large.pth'),
    'xciformer_small_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_small_384.pth',
                            input_size=(3, 384, 384), crop_pct=1.0),
    'xciformer_base_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_base_384.pth',
                            input_size=(3, 384, 384), crop_pct=1.0),
    'xciformer_large_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/xciformer/xciformer_large_384.pth',
                            input_size=(3, 384, 384), crop_pct=1.0),
}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

from .patches import *
from .mixers import *
from .blocks import *

class CrossCovarianceInceptionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=None, depths=None,
                num_heads=None, mlp_ratio=4., qkv_bias=True, 
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                act_layer=None, weight_init='',
                attention_heads=None,
                use_layer_scale=False, layer_scale_init_value=1e-5, 
                checkpoint_path=None,
                **kwargs, 
                ):
        
        super().__init__()
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)
        
        self.num_classes = num_classes
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.patch_embed = FirstPatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[0]))
        self.blocks1 = nn.ModuleList([
            XCiFormerBlock(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=2,)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(0, st2_idx)])


        self.patch_embed2 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[1]))
        self.blocks2 = nn.ModuleList([
            XCiFormerBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=2,)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, channel_layer_scale=channel_layer_scale,
                # )
            for i in range(st2_idx,st3_idx)])
        
        self.patch_embed3 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[2]))
        self.blocks3= nn.ModuleList([
            XCiFormerBlock(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                )
            for i in range(st3_idx, st4_idx)])
        
        self.patch_embed4 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[3]))
        self.blocks4 = nn.ModuleList([
            XCiFormerBlock(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                )
            for i in range(st4_idx,depth)])
        
        self.norm = norm_layer(embed_dims[-1])
        # Classifier head(s)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # set post block, for example, class attention layers

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
    
    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W) 
        # x = self.blocks1(x)
        for blk1 in self.blocks1:
            x = blk1(x, H, W)


        x = x.permute(0, 3, 1, 2)       
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W) 
        # x = self.blocks2(x)
        for blk2 in self.blocks2:
            x = blk2(x, H, W)
        
        x = x.permute(0, 3, 1, 2)  
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W) 
        # x = self.blocks3(x)
        for blk3 in self.blocks3:
            x = blk3(x, H, W)
        
        x = x.permute(0, 3, 1, 2)  
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W) 
        # x = self.blocks4(x)
        for blk4 in self.blocks4:
            x = blk4(x, H, W)
        x = x.flatten(1,2)

        x = self.norm(x)
        return x.mean(1)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


    
@register_model
def xciformer_small(pretrained=False, **kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3, 3, 9, 3]
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3
    
    model = CrossCovarianceInceptionTransformer(img_size=224,
        depths=depths,
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_small']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def xciformer_small_384(pretrained=False, **kwargs):
    depths = [3, 3, 9, 3]
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3
    
    model = CrossCovarianceInceptionTransformer(img_size=384,
        depths=depths,
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_small_384']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model




@register_model
def xciformer_base(pretrained=False, **kwargs):
    """ 
    47.866M  9.379G  84.598
    """
    depths = [4, 6, 14, 6]
    embed_dims = [96, 192, 384, 512]
    num_heads = [3, 6, 12, 16]
    attention_heads = [1]*4 + [3]*6 + [8] * 7 + [10] * 7 + [15] * 6
    
    model = CrossCovarianceInceptionTransformer(img_size=224,
        depths=depths, 
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_base']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def xciformer_base_384(pretrained=False, **kwargs):
    depths = [4, 6, 14, 6]
    embed_dims = [96, 192, 384, 512]
    num_heads = [3, 6, 12, 16]
    attention_heads = [1]*4 + [3]*6 + [8] * 7 + [10] * 7 + [15] * 6
    
    model = CrossCovarianceInceptionTransformer(img_size=384,
        depths=depths, 
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_base_384']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def xciformer_large(pretrained=False, **kwargs):
    """ 
    86.637M  14.048G 84.752
    """   
    depths = [4, 6, 18, 8]
    embed_dims = [96, 192, 448, 640]
    num_heads = [3, 6, 14, 20]
    attention_heads = [1]*4 + [3]*6 + [10] * 9 + [12] * 9 + [19] * 8
    
    model = CrossCovarianceInceptionTransformer(img_size=224,
        depths=depths, 
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_large']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def xciformer_large_384(pretrained=False, **kwargs): 
    depths = [4, 6, 18, 8]
    embed_dims = [96, 192, 448, 640]
    num_heads = [3, 6, 14, 20]
    attention_heads = [1]*4 + [3]*6 + [10] * 9 + [12] * 9 + [19] * 8
    
    model = CrossCovarianceInceptionTransformer(img_size=384,
        depths=depths, 
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True, layer_scale_init_value=1e-6, 
        **kwargs)
    model.default_cfg = default_cfgs['xciformer_large_384']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
