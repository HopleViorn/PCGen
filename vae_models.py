import torch
import torch.nn as nn
from VecSetX.vecset.models import autoencoder as vecset_ae
from hy3dshape.hy3dshape.models.autoencoders import ShapeVAE as HYVAE

class ShapeVAE(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder_type = args.vae_encoder_type
        self.device = device
        self.vae_encoder = self._load_encoder(args)

    def _load_encoder(self, args):
        if self.encoder_type == 'vecset':
            print("Loading VecSetX VAE Encoder...")
            encoder = vecset_ae.__dict__['point_vec1024x32_dim1024_depth24_nb'](pc_size=8192)
            checkpoint = torch.load(args.vecset_vae_weights, map_location='cpu')
            model_state_dict = checkpoint['model']
            if any(key.startswith('module.') for key in model_state_dict):
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            encoder.load_state_dict(model_state_dict)
            encoder = encoder.to(self.device).eval()
            return encoder
        elif self.encoder_type == 'hy3dshape':
            print("Loading hy3dshape ShapeVAE Encoder...")
            encoder = HYVAE.from_pretrained(
                'tencent/Hunyuan3D-2.1',
                subfolder='hunyuan3d-vae-v2-1',
                # device=self.device, # DataParallel will handle device placement
                dtype=torch.float16
            )
            encoder.eval()
            return encoder
        else:
            raise ValueError(f"Unknown VAE encoder type: {self.encoder_type}")

    def get_embedding(self, point_cloud):
        with torch.no_grad():
            if self.encoder_type == 'vecset':
                latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                condition = self.vae_encoder.learn(latent_embedding)
                return condition
            elif self.encoder_type == 'hy3dshape':
                latents = self.vae_encoder.encode(point_cloud, sample_posterior=True)
                condition = self.vae_encoder.decode(latents)
                return condition
    
    def get_latent(self, point_cloud):
        with torch.no_grad():
            if self.encoder_type == 'vecset':
                latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                return latent_embedding
            elif self.encoder_type == 'hy3dshape':
                latents = self.vae_encoder.encode(point_cloud, sample_posterior=True)
                return latents
    
    def forward(self, point_cloud):
        return self.get_embedding(point_cloud)

    def decode(self,latents):
        return self.vae_encoder.decode(latents)