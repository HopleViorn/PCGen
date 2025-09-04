from autoencoders import ShapeVAE

import torch

vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    subfolder='hunyuan3d-vae-v2-1',
    device='cuda',
    dtype=torch.float16
)


print(vae)