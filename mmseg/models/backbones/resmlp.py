import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.drop import DropPath
from mmcv.runner import BaseModule, ModuleList
from ..builder import BACKBONES
from .vit import VisionTransformer



class Affine(BaseModule):
    def __init__(self, embed_dims):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embed_dims))
        self.beta = nn.Parameter(torch.zeros(embed_dims))

    def forward(self, x):
        return self.alpha * x + self.beta    



class LayerScalePatchMLP(BaseModule):
    """
    LayerScalePatchMLP replaces the AttentionLayer used in LayerScaleTransformer by MLP,
    and replaces all LayerNorms as AffineLayers.
    It is a simple residual network that alternates: 
    1. a linear layer in which image PATCHES interact, independently and identically 
    across channels, 
    2. a two-layer feed-forward network in which CHANNELS interact independently per
    patch.
    
    """

    def __init__(self,
                 embed_dims,
                 mlp_ratio=4,
                 num_fcs=2,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 init_values=1e-4,
                 num_patches = 196):
        super(LayerScalePatchMLP, self).__init__()
        
        self.norm1 = Affine(embed_dims)
        
        self.attn = nn.Linear(num_patches, num_patches)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = Affine(embed_dims)
        
        self.ffn = FFN(
            embed_dims = embed_dims,
            feedforward_channels = mlp_ratio*embed_dims,
            num_fcs = num_fcs,
            ffn_drop=drop,
            dropout_layer=None,
            act_cfg=act_cfg,
            )  
        
        self.gamma_1 = nn.Parameter(init_values * torch.ones((embed_dims)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((embed_dims)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x 



@BACKBONES.register_module() 
class LayerScaleResMLP(VisionTransformer):

    def __init__(self, 
                 init_scale=1e-4,
                 img_size=(224, 224), 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=150, 
                 embed_dims=768, 
                 depth=12,
                 drop_rate=0.,
                 act_cfg=dict(type="GELU"),
                 drop_path_rate=0.0,
                 **kwargs):
        super(LayerScaleResMLP, self).__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            num_classes=num_classes, 
            embed_dims=embed_dims, 
            depth=depth,
            drop_rate=drop_rate,
            act_cfg=act_cfg,
            drop_path_rate=drop_path_rate,
            **kwargs
            )

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        dpr = [drop_path_rate for i in range(depth)]
        self.layers = ModuleList()
        for i in range(depth):
            self.layers.append(
                LayerScalePatchMLP(
                    embed_dims=embed_dims,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    act_cfg=act_cfg,
                    init_values=init_scale,
                    num_patches=num_patches)
                )


        self.norm1 = Affine(embed_dims)  # ViT uses LayerNorm
