# Modified from https://github.com/facebookresearch/deit/blob/c890ceb9303468cf47553da8764d7febabd9df68/cait_models.py

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import MultiheadAttention
from ..builder import BACKBONES
from .vit import VisionTransformer


class Attention_talking_head(BaseModule):
    # taken from https://github.com/facebookresearch/deit/blob/main/cait_models.py
    # with slight modifications (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()  # (3,B,n_head,n_q,n_head_dim)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = (q @ k.transpose(-2, -1))   # (B,n_head,n_q,n_k)
        
        # talking head: use linear combination of different heads (B,n_q,n_k,n_head) → (B,n_q,n_k,n_head)
        attn = self.proj_l(attn.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()  
                
        attn = attn.softmax(dim=-1)  # convert from logits to probs
        
        # talking head: use linear in head-dimension again
        attn = self.proj_w(attn.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()  
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class LayerScale_Block(BaseModule):
    """
    In LayerScale, each embed_dim in the output of the residual block has a weight, which is 
    given by a learnable vector (namely gamma_1 and gamma_2).
    """
    
    def __init__(self,
                 embed_dims,
                 num_heads,
                 init_scale=1e-4,
                 qkv_bias=False, 
                 qk_scale=None,
                 drop=0.,
                 attn_drop_rate=0.,
                 drop_path=0.,
                 mlp_ratio=4.,
                 num_fcs=2,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 batch_first=True):
        super(LayerScale_Block, self).__init__()
        
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        
        self.attn = Attention_talking_head(
            embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop,
            qkv_bias=qkv_bias)
        
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        
        # Note: default：drop_prob = 0.0 in cait and 0.1 in mmcv
        self.dropout_layer = build_dropout(  
            dict(type='DropPath', drop_prob=drop_path)) if drop_path > 0. else nn.Identity()
        
        feedforward_channels = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims = embed_dims,
            feedforward_channels = feedforward_channels,
            num_fcs = num_fcs,
            ffn_drop=drop,
            dropout_layer=None,
            act_cfg=act_cfg,
            )  
        
        # LayerScale:
        self.gamma_1 = nn.Parameter(init_scale * torch.ones((embed_dims)),requires_grad=True)  # for atten
        self.gamma_2 = nn.Parameter(init_scale * torch.ones((embed_dims)),requires_grad=True)  # for ffn


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)


    @property
    def norm2(self):
        return getattr(self, self.norm2_name)


    def forward(self, x):        
        x = x + self.dropout_layer(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.dropout_layer(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    
    
@BACKBONES.register_module()    
class LayerScaleTransformer(VisionTransformer):
    def __init__(self,
                 num_layers=12,
                 drop_path_rate=0., 
                 embed_dims=768, 
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 init_scale=1e-4, 
                 **kwargs):
        super(LayerScaleTransformer, self).__init__(
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)


        dpr = [drop_path_rate for i in range(num_layers)] 
        
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                LayerScale_Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop_rate, 
                    attn_drop_rate=attn_drop_rate,
                    drop_path=dpr[i], 
                    norm_cfg=norm_cfg, 
                    act_cfg=act_cfg,
                    init_scale=init_scale))

