# Inspired: https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/models.py
import torch.nn as nn
from functools import partial

class MappingNetwork(nn.Module):
    def __init__(
        self,
        clip_dim,
        sam_dim,
        out_dim,
        h=4096,
        n_blocks=4,
        norm_type='ln',
        act_first=False
    ):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        
        self.lin_clip_img = nn.Sequential(
            nn.Linear(clip_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.3)
        )
        
        self.lin_clip_txt = nn.Sequential(
            nn.Linear(clip_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.3)
        )
        
        self.lin_sam = nn.Sequential(
            nn.Linear(sam_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.3)
        )
        
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        
        self.final_lin = nn.Linear(h, out_dim)
        
        self.n_blocks = n_blocks
        
        # weight init for future
        # should we weight clip img, clip txt, sam (?)
        
    def forward(self, clip_img, clip_txt, sam_feat):
        clip_i = self.lin_clip_img(clip_img)
        clip_t = self.lin_clip_txt(clip_txt)
        sam_f = self.lin_sam(sam_feat)
        
        x = clip_i + clip_t + sam_f
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
            
        x = self.final_lin(x)
        return x
        
        