from torch.utils.checkpoint import checkpoint
from src.climax_core.utils.pos_embed import interpolate_pos_embed
import torch
import torch.nn as nn
import sys
import os
from src.climax_core.arch import ClimaX


class PM25Model(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        print("Initializing PM25Model...")
        print(f"Device: {device}")
        print(f"Checkpoint: {config['data']['checkpoint_path']}")
        
        # Variables
        self.variables = config['data']['variables']
        print(f"Input variables: {self.variables}")
        
        # ClimaX backbone
        print("Building ClimaX backbone...")
        model_config = config['model']
        self.climax = ClimaX(
            default_vars=self.variables,
            img_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            embed_dim=model_config['embed_dim'],
            depth=model_config['depth'], 
            decoder_depth=model_config['decoder_depth'],
            num_heads=model_config['num_heads'], 
            mlp_ratio=model_config['mlp_ratio'],
            drop_path=model_config['drop_path'], 
            drop_rate=model_config['drop_rate']
        )
        print("ClimaX backbone created")
        
        # Load pretrained weights (except head and pos_embed)
        print("Loading pretrained weights...")
        ckpt = torch.load(config['data']['checkpoint_path'], map_location='cpu')['state_dict']
        sd = {k[4:]:v for k,v in ckpt.items() if k.startswith('net.')}
        sd = {k:v for k,v in sd.items() if not k.startswith('head.')}
        
        # Remove pos_embed from checkpoint - let it initialize randomly for new resolution
        if 'pos_embed' in sd:
            print(f"Skipping pos_embed from checkpoint (different resolution)")
            del sd['pos_embed']
        
        self.climax.load_state_dict(sd, strict=False)
        print(f"{len(sd)} weights loaded from checkpoint")
        
        # Freeze encoder
        frozen_params = 0
        for p in self.climax.parameters():
            p.requires_grad = False
            frozen_params += p.numel()
        print(f"Encoder frozen ({frozen_params:,} parameters)")
        
        # PM2.5 regression head
        embed_dim = model_config['embed_dim']
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 1, 1)
        )
        trainable_params = sum(p.numel() for p in self.head.parameters())
        print(f"Regression head created ({trainable_params:,} trainable parameters)")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.climax = nn.DataParallel(self.climax)
            self.head = nn.DataParallel(self.head)
        self.to(device)
        print("PM25Model initialized successfully")

    def forward(self, x, lead_times):
        B,C,H,W = x.shape
        device = x.device
        lt = torch.tensor(lead_times, dtype=torch.float32, device=device)
        feats = checkpoint(self.climax.module.forward_encoder, x, lt, self.variables) if hasattr(self.climax, "module") else checkpoint(self.climax.forward_encoder, x, lt, self.variables)
        for blk in (self.climax.module.blocks if hasattr(self.climax, "module") else self.climax.blocks):
            feats = blk(feats)
        p = (self.climax.module.patch_size if hasattr(self.climax, "module") else self.climax.patch_size)
        h, w = H // p, W // p
        num = h * w
        feats = feats[:, :num]\
            .reshape(B, h, w, -1)\
            .permute(0, 3, 1, 2)
        if feats.shape[-2:] != (H, W):
            feats = nn.functional.interpolate(
                feats, size=(H, W), mode='bilinear', align_corners=False
            )
        return self.head(feats).squeeze(1)
