import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
import numpy as np

from .attention_processors import CrossAttentionProcessor
from ...utils import logger

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
        self, *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                logger.info('Save kv cache,this should be called only once for one mesh')
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type
        self.count = 0

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def set_default_cross_attention_processor(self):
        self.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ

import torch
import torch.nn as nn
from typing import List

# 这是一个标准的MLP层，使用Softplus作为激活函数
class SoftplusLayer(nn.Module):
    """一个标准的MLP层，包含一个线性层和一个Softplus激活函数"""
    def __init__(self, dim_in, dim_out, use_bias=True, activation=None):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        # 如果提供了激活函数则使用它，否则默认使用Softplus
        self.activation = nn.Softplus() if activation is None else activation

    def forward(self, x):
        return self.activation(self.linear(x))

# 这是与SirenNet结构对齐的Softplus网络
class SoftplusNet(nn.Module):
    """
    一个与SirenNet具有相同结构定义（层数、隐藏维度、跳跃连接）的MLP，
    但使用Softplus作为激活函数，用于实验对比。
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip=[], skip_dim=3):
        """
        初始化函数。参数与SirenNet保持一致，但不包含w0等SIREN专属参数。
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]
        self.skip_dim = skip_dim

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            
            if is_first:
                layer_dim_in = dim_in
            else:
                layer_dim_in = dim_hidden

            # 如果当前层需要跳跃连接，则将输入维度加上跳跃输入的维度
            if self.skip[ind]:
                layer_dim_in += skip_dim

            self.layers.append(SoftplusLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                use_bias=True
            ))
            
        # 最后一层通常是线性的，不带激活函数（或使用恒等激活）
        # 这里直接使用一个线性层，等同于SoftplusLayer(activation=nn.Identity())
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, skip_input=None):
        """
        前向传播逻辑，与SirenNet完全相同。
        """
        # 保存原始输入，用于后续的跳跃连接
        original_x = x
        if skip_input is None:
            skip_input = original_x

        for k, layer in enumerate(self.layers):
            # 如果当前层需要跳跃连接
            if self.skip[k] and skip_input is not None:
                # 将当前特征与跳跃输入拼接
                x = layer(torch.cat((x, skip_input), dim=-1))
            else:
                x = layer(x)
                
        return self.last_layer(x)

class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0*x)

class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip = [], w0 = 5., w0_initial = 5., activation = None, skip_dim=3):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]
        self.skip_dim = skip_dim

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            
            if is_first:
                layer_dim_in = dim_in
            else:
                layer_dim_in = dim_hidden

            if self.skip[ind]:
                layer_dim_in += skip_dim

            self.layers.append(SirenLayer(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = True,
                is_first = is_first,
                activation = activation
            ))
        self.last_layer = SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = True, activation = nn.Identity())

    def forward(self, x, skip_input=None):
        
        for k,layer in enumerate(self.layers):
            if self.skip[k] and skip_input is not None:
                x = layer(torch.concat((x, skip_input), dim=-1))
            else:
                x = layer(x)
        return self.last_layer(x)

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class CrossAttentionSirenDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        siren_num_layers: int = 3,
        siren_w0: float = 30.0
    ):
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)

        # TODO:self.siren =
        # self.siren = SirenNet(
        #     dim_in=width + 3,
        #     dim_hidden=128,
        #     dim_out=out_channels,
        #     num_layers=3,
        #     w0=siren_w0,
        #     w0_initial=siren_w0
        # )
        # self.siren = SoftplusNet(
        #     dim_in=width,
        #     dim_hidden=128,
        #     dim_out=out_channels,
        #     num_layers=5,
        #     skip=[0,3],
        # )

        self.siren = SirenNet(
            dim_in=width,
            dim_hidden=128,
            dim_out=out_channels,
            num_layers=5,
            skip=[0,3],
            w0=siren_w0,
            w0_initial=siren_w0
        )


        self.label_type = label_type
        self.count = 0

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def set_default_cross_attention_processor(self):
        self.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)

        q_for_grad = queries.clone().detach().requires_grad_(True)

        sdf_v = self.siren(x, skip_input=q_for_grad)
        sdf_grad = gradient(sdf_v, q_for_grad)

        return sdf_v, sdf_grad

def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    src = src.float()
    from torch_cluster import fps as fps_fn
    output = fps_fn(src, batch, ratio, random_start, batch_size, ptr)
    return output


class PointCrossAttentionEncoder(nn.Module):

    def __init__(
        self, *,
        num_latents: int,
        downsample_ratio: float,
        pc_size: int,
        pc_sharpedge_size: int,
        fourier_embedder: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        normal_pe: bool = False,
        qkv_bias: bool = True,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
        qk_norm: bool = False
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        if pc_sharpedge_size == 0:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is zero')
        else:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is given, using pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}')

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size

        self.fourier_embedder = fourier_embedder

        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                n_ctx=num_latents,
                width=width,
                layers=layers,
                heads=heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm
            )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def sample_points_and_latents(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        B, N, D = pc.shape
        num_pts = self.num_latents * self.downsample_ratio

        # Compute number of latents
        num_latents = int(num_pts / self.downsample_ratio)

        # Compute the number of random and sharpedge latents
        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        # Split random and sharpedge surface points
        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[
                   1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        # Randomly select random surface points and random query points
        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        random_query_ratio = num_random_query / input_random_pc_size
        idx_random_pc = torch.randperm(random_pc.shape[1], device=random_pc.device)[:input_random_pc_size]
        input_random_pc = random_pc[:, idx_random_pc, :]
        flatten_input_random_pc = input_random_pc.view(B * input_random_pc_size, D)
        N_down = int(flatten_input_random_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps(flatten_input_random_pc, batch_down, ratio=random_query_ratio)
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        # Randomly select sharpedge surface points and sharpedge query points
        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype).to(pc.device)
            query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_random_pc.dtype).to(pc.device)
        else:
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1], device=sharpedge_pc.device)[
                               :input_sharpedge_pc_size]
            input_sharpedge_pc = sharpedge_pc[:, idx_sharpedge_pc, :]
            flatten_input_sharpedge_surface_points = input_sharpedge_pc.view(B * input_sharpedge_pc_size, D)
            N_down = int(flatten_input_sharpedge_surface_points.shape[0] / B)
            batch_down = torch.arange(B).to(pc.device)
            batch_down = torch.repeat_interleave(batch_down, N_down)
            idx_query_sharpedge = fps(flatten_input_sharpedge_surface_points, batch_down, ratio=sharpedge_query_ratio)
            query_sharpedge_pc = flatten_input_sharpedge_surface_points[idx_query_sharpedge].view(B, -1, D)

        # Concatenate random and sharpedge surface points and query points
        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        # PE
        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        # Concat normal if given
        if self.point_feats != 0:

            random_surface_feats, sharpedge_surface_feats = torch.split(feats, [self.pc_size, self.pc_sharpedge_size],
                                                                        dim=1)
            input_random_surface_feats = random_surface_feats[:, idx_random_pc, :]
            flatten_input_random_surface_feats = input_random_surface_feats.view(B * input_random_pc_size, -1)
            query_random_feats = flatten_input_random_surface_feats[idx_query_random].view(B, -1,
                                                                                           flatten_input_random_surface_feats.shape[
                                                                                               -1])

            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_feats = torch.zeros(B, 0, self.point_feats,
                                                            dtype=input_random_surface_feats.dtype).to(pc.device)
                query_sharpedge_feats = torch.zeros(B, 0, self.point_feats, dtype=query_random_feats.dtype).to(
                    pc.device)
            else:
                input_sharpedge_surface_feats = sharpedge_surface_feats[:, idx_sharpedge_pc, :]
                flatten_input_sharpedge_surface_feats = input_sharpedge_surface_feats.view(B * input_sharpedge_pc_size,
                                                                                           -1)
                query_sharpedge_feats = flatten_input_sharpedge_surface_feats[idx_query_sharpedge].view(B, -1,
                                                                                                        flatten_input_sharpedge_surface_feats.shape[
                                                                                                            -1])

            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)
            input_feats = torch.cat([input_random_surface_feats, input_sharpedge_surface_feats], dim=1)

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)

        # print(f'query_pc: {query_pc.shape}')
        # print(f'input_pc: {input_pc.shape}')
        # print(f'query_random_pc: {query_random_pc.shape}')
        # print(f'input_random_pc: {input_random_pc.shape}')
        # print(f'query_sharpedge_pc: {query_sharpedge_pc.shape}')
        # print(f'input_sharpedge_pc: {input_sharpedge_pc.shape}')

        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1]), [query_pc, input_pc,
                                                                                      query_random_pc, input_random_pc,
                                                                                      query_sharpedge_pc,
                                                                                      input_sharpedge_pc]

    def forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        query, data, pc_infos = self.sample_points_and_latents(pc, feats)

        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data

        latents = self.cross_attn(query, data)
        if self.self_attn is not None:
            latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc_infos