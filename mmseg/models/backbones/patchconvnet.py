# Take from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/
# and https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py
# with some modification for segmentation task.

import torch
import torch.nn as nn
from functools import partial
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn import build_activation_layer
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init, trunc_normal_)
from mmcv.utils import to_2tuple
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


class SqueezeExcite(BaseModule):
    # Take from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet_blocks.py
    # with slight modification.
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, 
            in_chs, 
            rd_ratio=0.25, 
            rd_channels=None, 
            act_layer=dict(type='ReLU', inplace=True),
            gate_layer=nn.Sigmoid(), 
            force_act_layer=None,
            rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = build_activation_layer(act_layer)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)



class Learned_Aggregation_Layer(nn.Module):
    """Learned_Aggregation_Layer uses cls_token to choose which patches the net should pay more attention to.
    Take from https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py 
    with slight modification.
    Specifically, to deal with segmentation problems, the cls_token is used as query to 
    calculate the attention vector with all patches, which yield an output with shape of
    (B, 1, num_patches). Each element of this attention vector is furtherly used as the 
    weight of the corresponding channel value matrix.
    
    """
    def __init__(self, 
                 dim, 
                 num_heads=1,
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x (B, 1 + num_patches, E)
        B, N, C = x.shape
        N -= 1  # minus the num of cls_token
        q = self.q(x[:, 0]).unsqueeze(1).reshape(
            B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q (B, num_heads, 1, E_per_head)
        
        k = self.k(x[:, 1:]).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        # k (B, num_heads, num_patches, E_per_head)

        q = q * self.scale
        
        v = self.v(x[:, 1:]).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3).contiguous()
        # v (B, num_heads, num_patches, E_per_head)

        attn = (q @ k.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.transpose(-2, -1).contiguous()
        out = (attn * v).transpose(1, 2).contiguous().reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class Layer_scale_init_Block_only_token(nn.Module):
    """
    Layer_scale_init_Block_only_token uses Learned_Aggregation_Layer to gather the info
    for classification or to learn to focus on some patches, which is typically used in
    the last layer of the backbone.
    """
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Attention_block=Learned_Aggregation_Layer,
                 FFN_block=FFN,
                 init_values=1e-4):
                 
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer,
            drop_layer=None, 
            drop=drop)
            
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x_cls)))
        return x_cls


class Conv_blocks_se(nn.Module):
    """
    Conv_blocks_se consist of a 3x3ConvLayer and a SElayer, and 2 1x1ConvLayers are added
    in the begining and the end.
    """
    def __init__(self, dim):
        super().__init__()

        self.qkv_pos = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim, 
                groups=dim,  # unusual
                kernel_size=3,
                padding=1, 
                stride=1, 
                bias=True),
            nn.GELU(),
            SqueezeExcite(dim, rd_ratio=0.25),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        # H = W = int(N**0.5) # This may cause a bug if height is not equal to width.
        H = hw_shape[0]
        W = hw_shape[1]
        x = x.transpose(-1, -2).contiguous()
        x = x.reshape(B, C, H, W)
        x = self.qkv_pos(x)
        x = x.reshape(B, C, N)
        x = x.transpose(-1, -2).contiguous()
        return x


class Layer_scale_init_Block(nn.Module):
    """
    The block of the main part of the backbone, which can be seemed as the 
    AttentionLayer in the TransformerLayer, the differences are two aspects:
    1. attn layer used here is SE block instead of MultiHeadAttention.
    2. FFN layer is not used here.
    """
    def __init__(self, 
                 dim, 
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Attention_block=None, 
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, hw_shape):
        # SE and Layerscale are used together here?
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), hw_shape))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
    )


class ConvStem(nn.Module):
    """ Image to Patch Embedding
    Modification: returns the shape for transforming the tokens from (B, C, E)
    back to the image (B, H, W, E) later.
    """

    def __init__(self, 
                 img_size=224,
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])   # this img_size is for testing
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = torch.nn.Sequential(
            conv3x3(3, embed_dim // 8, 2),
            nn.GELU(),
            conv3x3(embed_dim // 8, embed_dim // 4, 2),
            nn.GELU(),
            conv3x3(embed_dim // 4, embed_dim // 2, 2),
            nn.GELU(),
            conv3x3(embed_dim // 2, embed_dim, 2),
        )

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, hw_shape


@BACKBONES.register_module()
class PatchConvnet(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768, 
                 depth=12,
                 num_heads=6,
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path=0.,
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm,
                 global_pool=None,
                 Patch_layer=ConvStem, 
                 act_layer=nn.GELU,
                 dpr_constant=True,
                 init_scale=1e-4,
                 block_layers=Layer_scale_init_Block,  # Layer_scale_init_Block uses SE as attention layer.
                 Attention_block=Conv_blocks_se,  # Note: some Conv2ds are added.
                 block_layers_token=Layer_scale_init_Block_only_token,  # uses Learned_Aggregation_Layer to focus on some patches.
                 Attention_block_token_only=Learned_Aggregation_Layer,
                 Mlp_block_token_only=FFN,
                 depth_token_only=1,
                 mlp_ratio_clstk=3.0,
                 multiclass=False,
                 pretrained=None,
                 init_cfg=None,
                 out_indices=-1):
        super(PatchConvnet, self).__init__(init_cfg = init_cfg)
        
        self.out_indices = out_indices
        
        self.multiclass = multiclass
        self.patch_size = patch_size
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) * (img_size // patch_size)                 
        
        # Since this is acturally a CNN network, pos_embeddings are not used.
        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)


        if not dpr_constant:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate for i in range(depth)]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=int(embed_dim),
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio_clstk,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=drop_path, 
                norm_layer=norm_layer,
                act_layer=act_layer,
                Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, 
                init_values=init_scale)
            for i in range(depth_token_only)])

        self.norm = norm_layer(int(embed_dim))

        self.total_len = depth_token_only+depth
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
            
        self.pretrained = pretrained


    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict, False)
        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501

            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def get_num_layers(self):
        return len(self.blocks)


    def forward(self, x):
        
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        outs = []
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, hw_shape)
            if i in self.out_indices:
                out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()

                outs.append(out)
                
                
        for _i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
            
        x = self.norm(x)
        B, _, C = x.shape
        out = x.reshape(B, hw_shape[0], hw_shape[1],
                          C).permute(0, 3, 1, 2).contiguous()
        outs.append(out)
            
        return tuple(outs)


        
        
def S60(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384, 
        depth=60, 
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,
        **kwargs)
    return model


