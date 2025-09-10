import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder
from diffusers.models.unets.unet_1d_blocks import ResConvBlock, SelfAttention1d, get_down_block, get_up_block, Upsample1d
from diffusers.models.attention_processor import SpatialNorm

from hy3dshape.hy3dshape.models.autoencoders import ShapeVAE as HYVAE

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)


def sincos_embedding(input, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param input: a N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim //2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /half
    ).to(device=input.device)
    for _ in range(len(input.size())):
        freqs = freqs[None]
    args = input.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CondSurfPosNet(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """

    def __init__(self, use_cf, cond_dim=64):
        super(CondSurfPosNet, self).__init__()
        self.embed_dim = 768
        self.use_cf = use_cf

        layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True,
                                                   dim_feedforward=1024, dropout=0.1)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True,
                                                   dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerDecoder(layer, 12, nn.LayerNorm(self.embed_dim))

        # self.cond_mapper = nn.Linear(cond_dim, self.embed_dim)
        # self.cond_decoder = nn.TransformerEncoder(enc_layer, 12, nn.LayerNorm(self.embed_dim))
        self.vae_decoder = HYVAE.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            subfolder='hunyuan3d-vae-v2-1',
            dtype=torch.float16
        )
        self.vae_decoder.eval() # Set to eval mode
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        self.cond_projector = nn.Linear(1024, self.embed_dim)

        self.p_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6),
        )

        if self.use_cf:
            self.class_embed = Embedder(11, self.embed_dim)

        return

       
    def forward(self, surfPos, timesteps, class_label, condition, is_train=False):
        """ forward pass """
        bsz = timesteps.size(0)
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        p_embeds = self.p_embed(surfPos)
    
        if self.use_cf:  # classifier-free
            if is_train:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(bsz,1) <= 0.1
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label)
            tokens = p_embeds + time_embeds + c_embeds
        else:
            tokens = p_embeds + time_embeds
        
        # condition = self.cond_mapper(condition) #(4096 x 1024)
        with torch.no_grad():
            condition_decoded = self.vae_decoder.decode(condition.to(torch.float16))
        condition_decoded = self.cond_projector(condition_decoded.to(torch.float32))
        output = self.net(tgt=tokens.permute(1,0,2), memory=condition_decoded.permute(1,0,2)).transpose(0,1)
        # output = self.net(tgt=tokens.permute(1,0,2), memory=condition.permute(1,0,2)).transpose(0,1)
        pred = self.fc_out(output)

        return pred
    

class CondSurfZNet(nn.Module):
    """
    Transformer-based latent diffusion model for surface position
    """
    def __init__(self, use_cf, cond_dim=1024):
        super(CondSurfZNet, self).__init__()
        self.embed_dim = 768
        self.use_cf = use_cf

        layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True,
                                                   dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerDecoder(layer, 12, nn.LayerNorm(self.embed_dim))

        self.cond_mapper = nn.Linear(cond_dim, self.embed_dim)

        self.z_embed = nn.Sequential(
            nn.Linear(3*16, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.p_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ) 

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 3*16),
        )

        if self.use_cf:
            self.class_embed = Embedder(11, self.embed_dim)

        return

       
    def forward(self, surfZ, timesteps, surfPos, surf_mask, class_label, condition, is_train=False):
        """ forward pass """
        bsz = timesteps.size(0)

        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1) 
        z_embeds = self.z_embed(surfZ) 
        p_embeds = self.p_embed(surfPos)

        if self.use_cf:  # classifier-free
            if is_train:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(bsz,1) <= 0.1  
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label) 
            tokens = z_embeds + p_embeds + time_embeds + c_embeds
        else:
            tokens = z_embeds + p_embeds + time_embeds

        condition = self.cond_mapper(condition)
        output = self.net(
            tgt=tokens.permute(1,0,2),
            memory=condition.permute(1,0,2),
            tgt_key_padding_mask=surf_mask,
        ).transpose(0,1)
        
        pred = self.fc_out(output)
        return pred
    
class CondSurfPosNetFM(CondSurfPosNet):
    def forward(self, surfPos, timesteps, class_label, condition, is_train=False):
        """ forward pass """
        bsz = timesteps.size(0)
        
        # Convert timesteps to float and scale for sincos_embedding
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        p_embeds = self.p_embed(surfPos)
    
        if self.use_cf:  # classifier-free
            if is_train:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(bsz,1) <= 0.1  
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label) 
            tokens = p_embeds + time_embeds + c_embeds
        else:
            tokens = p_embeds + time_embeds
        
        condition = self.cond_mapper(condition)
        output = self.net(tgt=tokens.permute(1,0,2), memory=condition.permute(1,0,2)).transpose(0,1)
        pred = self.fc_out(output)

        return pred

class CondSurfZNetFM(CondSurfZNet):
    def forward(self, surfZ, timesteps, surfPos, surf_mask, class_label, condition, is_train=False):
        """ forward pass """
        bsz = timesteps.size(0)

        # Convert timesteps to float and scale for sincos_embedding
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        z_embeds = self.z_embed(surfZ)
        p_embeds = self.p_embed(surfPos)

        if self.use_cf:  # classifier-free
            if is_train:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(bsz,1) <= 0.1  
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label) 
            tokens = z_embeds + p_embeds + time_embeds + c_embeds
        else:
            tokens = z_embeds + p_embeds + time_embeds

        condition = self.cond_mapper(condition)
        output = self.net(
            tgt=tokens.permute(1,0,2),
            memory=condition.permute(1,0,2),
            tgt_key_padding_mask=surf_mask,
        ).transpose(0,1)
        
        pred = self.fc_out(output)
        return pred