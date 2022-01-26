import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.drop import DropPath
from mmcv.runner import BaseModule, ModuleList
from ..builder import BACKBONES
from .vit import VisionTransformer
  



class MLP_Mixer_Layer(BaseModule):
    """
    MLP_Mixer_Layer replaces the AttentionLayer used in LayerScaleTransformer by MLP,
    and replaces all LayerNorms as AffineLayers.
    It is a simple residual network that alternates: 
    1. a linear layer in which image PATCHES interact, independently and identically 
    across channels, 
    2. a two-layer feed-forward network in which CHANNELS interact independently per
    patch.
    3. The differences between MLP-Mixer and ResMLP is that ResMLP uses layerscale and 
    replaces the Layernorm as Affine.
    
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
        super(MLP_Mixer_Layer, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dims)
        
        self.attn = nn.Linear(num_patches, num_patches)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(embed_dims)
        
        self.ffn = FFN(
            embed_dims = embed_dims,
            feedforward_channels = mlp_ratio*embed_dims,
            num_fcs = num_fcs,
            ffn_drop=drop,
            dropout_layer=None,
            act_cfg=act_cfg,
            )  

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x 



@BACKBONES.register_module() 
class MLP_Mixer(VisionTransformer):

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
        super(MLP_Mixer, self).__init__(
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


        self.norm1 = nn.LayerNorm(embed_dims)  # ViT uses LayerNorm
