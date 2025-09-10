import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from diffusers import AutoencoderKL, DDPMScheduler
from network import *
from network_conditional import CondSurfPosNet, CondSurfZNet
from VecSetX.vecset.models import autoencoder as vecset_ae
from vae_models import ShapeVAE


class SurfVAETrainer():
    """ Surface VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )

        # Load pretrained surface vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    
        
    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for surf_uv in self.train_dataloader:
            with torch.cuda.amp.autocast():
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE 
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, surf_uv) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, surf_uv).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_uv)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return


class EdgeVAETrainer():
    """ Edge VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL1D(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
            up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
            block_out_channels=[128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )

        # Load pretrained surface vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for edge_u in self.train_dataloader:
            with torch.cuda.amp.autocast():
                edge_u = edge_u.to(self.device).permute(0,2,1)
                self.optimizer.zero_grad() # zero gradient

               # Pass through VAE 
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample
                
                # Loss functions
                kl_loss =  0.5 * torch.sum(
                    torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar,
                    dim=[1, 2],
                ).mean()            
                mse_loss = loss_fn(dec, edge_u) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for edge_u in self.val_dataloader:
                edge_u = edge_u.to(self.device).permute(0,2,1)
                
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, edge_u).mean((1,2)).sum().item()
                total_loss += loss
                total_count += len(edge_u)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return
    

class SurfPosTrainer():
    """ Surface Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = SurfPosNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=self.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=32)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=32)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                if self.use_cf:
                    data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                    surfPos, class_label = data_cuda 
                else:
                    surfPos = data.to(self.device)
                    class_label = None

                bsz = len(surfPos)
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(surfPos_diffused, timesteps, class_label, True)
              
                # Compute loss
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        total_loss = [0,0,0,0,0]

        for data in self.val_dataloader:
            if self.use_cf:
                data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                surfPos, class_label = data_cuda 
            else:
                surfPos = data.to(self.device)
                class_label = None

            bsz = len(surfPos)
        
            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                with torch.no_grad():
                    surfPos_pred = self.model(surfPos_diffused, timesteps, class_label) 
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss[idx] += loss

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        return
    

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'surfpos_epoch_'+str(self.epoch)+'.pt'))
        return


class SurfZTrainer():
    """ Surface Latent Geometry Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = SurfZNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=32)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=32)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data_cuda = [x.to(self.device) for x in data] # map to gpu               
                if self.use_cf:
                    surfPos, surfPnt, surf_mask, class_label = data_cuda 
                else:
                    surfPos, surfPnt, surf_mask = data_cuda
                    class_label = None

                bsz = len(surfPos)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                conditions = [surfPos]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)  
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                surfPos = aug_data[0]

                # Pass through surface VAE to sample latent z 
                with torch.no_grad():
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)     

                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ.shape).to(self.device)  
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                # Predict noise
                surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, class_label, True)

                # Loss
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])        
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*5

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                surfPos, surfPnt, surf_mask, class_label = data_cuda 
            else:
                surfPos, surfPnt, surf_mask = data_cuda
                class_label = None
            bsz = len(surfPos)

            # Pass through surface VAE to sample latent z 
            with torch.no_grad():
                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z = self.surf_vae(surf_uv)
            surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)    
            tokens = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(tokens.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(tokens, noise, timesteps)
                with torch.no_grad():
                    pred = self.model(diffused, timesteps, surfPos, surf_mask, class_label)
                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        return
    

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'surfz_epoch_'+str(self.epoch)+'.pt'))
        return
    

class EdgePosTrainer():
    """ Edge Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = EdgePosNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data_cuda = [x.to(self.device) for x in data] # map to gpu
                if self.use_cf:
                    edgePos, surfPnt, surfPos, surf_mask, class_label = data_cuda
                else:
                    edgePos, surfPnt, surfPos, surf_mask = data_cuda
                    class_label = None
                bsz = len(surfPos)

                # Pass through surface VAE to sample latent z 
                with torch.no_grad():
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)     
                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

                # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                conditions = [surfPos, surfZ]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)  
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                surfPos, surfZ = aug_data[0], aug_data[1]

                # Zero gradient
                self.optimizer.zero_grad() 

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                edgePos_noise = torch.randn(edgePos.shape).to(self.device)  
                edgePos_diffused = self.noise_scheduler.add_noise(edgePos, edgePos_noise, timesteps)

                # Predict noise
                edgePos_pred = self.model(edgePos_diffused, timesteps, surfPos, surfZ, surf_mask, class_label, True)

                # Loss
                total_loss = self.loss_fn(edgePos_pred[~surf_mask], edgePos_noise[~surf_mask])
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*3

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                edgePos, surfPnt, surfPos, surf_mask, class_label = data_cuda
            else:
                edgePos, surfPnt, surfPos, surf_mask = data_cuda
                class_label = None
            bsz = len(surfPos)

            # Pass through surface VAE to sample latent z 
            with torch.no_grad():
                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z = self.surf_vae(surf_uv)
            surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)
            surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(edgePos.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(edgePos, noise, timesteps)
                with torch.no_grad():
                    pred = self.model(diffused, timesteps, surfPos, surfZ, surf_mask, class_label)
                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2]}, step=self.iters)
        return
    

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return
    

class EdgeZTrainer():
    """ Edge Latent Z Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = EdgeZNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        # Load pretrained edge vae (fast encode version)
        edge_vae = AutoencoderKL1DFastEncode(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
            up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
            block_out_channels=[128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )
        edge_vae.load_state_dict(torch.load(args.edgevae), strict=False)
        edge_vae = nn.DataParallel(edge_vae) # distributed inference 
        self.edge_vae = edge_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data_cuda = [x.to(self.device) for x in data] # map to gpu
                if self.use_cf:
                    edgePnt, edgePos, edge_mask, surfPnt, surfPos, vertPos, class_label = data_cuda
                else:
                    edgePnt, edgePos, edge_mask, surfPnt, surfPos, vertPos = data_cuda
                    class_label = None

                bsz = len(surfPos)

                # Pass through surface/edge VAE to sample latent z 
                with torch.no_grad():
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
    
                    edge_u = edgePnt.flatten(0,1).flatten(0,1).permute(0,2,1)
                    edge_z =self.edge_vae(edge_u)
                    edge_z = edge_z.unflatten(0, (-1, self.max_edge)).unflatten(0, (bsz,-1)).permute(0,1,2,4,3)

                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                edgeZ = edge_z.flatten(-2,-1)  * self.z_scaled 
                joint_data = torch.concat([edgeZ, vertPos], -1) # vertex as part of edge, 18-Dim total

                # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                conditions = [edgePos, surfPos, surfZ]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)  
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                edgePos, surfPos, surfZ = aug_data[0], aug_data[1], aug_data[2]

                # Zero gradient
                self.optimizer.zero_grad() 

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                joint_data_noise = torch.randn(joint_data.shape).to(self.device)  
                joint_data_diffused = self.noise_scheduler.add_noise(joint_data, joint_data_noise, timesteps)

                # Predict noise
                joint_data_pred = self.model(joint_data_diffused, timesteps, edgePos, surfPos, surfZ, edge_mask, class_label, True)

                # Loss
                total_loss = self.loss_fn(joint_data_pred[~edge_mask], joint_data_noise[~edge_mask])
                loss_z = self.loss_fn(joint_data_pred[~edge_mask][:,:12], joint_data_noise[~edge_mask][:,:12])
                loss_pos = self.loss_fn(joint_data_pred[~edge_mask][:,12:], joint_data_noise[~edge_mask][:,12:])
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss, "Loss-noise(z)": loss_z, "Loss-noise(v)": loss_pos}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item(), loss_z=loss_z.item(), loss_pos=loss_pos.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*3

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                edgePnt, edgePos, edge_mask, surfPnt, surfPos, vertPos, class_label = data_cuda
            else:
                edgePnt, edgePos, edge_mask, surfPnt, surfPos, vertPos = data_cuda
                class_label = None
            bsz = len(surfPos)

            # Pass through surface/edge VAE to sample latent z 
            with torch.no_grad():
                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z = self.surf_vae(surf_uv)
                surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
                edge_u = edgePnt.flatten(0,1).flatten(0,1).permute(0,2,1)
                edge_z =self.edge_vae(edge_u)
                edge_z = edge_z.unflatten(0, (-1, self.max_edge)).unflatten(0, (bsz,-1)).permute(0,1,2,4,3)
            surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
            edgeZ = edge_z.flatten(-2,-1)  * self.z_scaled 
            joint_data = torch.concat([edgeZ, vertPos], -1) # vertex as part of edge, 18D total

            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(joint_data.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(joint_data, noise, timesteps)
                with torch.no_grad():
                    pred = self.model(diffused, timesteps, edgePos, surfPos, surfZ, edge_mask, class_label)
                loss = mse_loss(pred[~edge_mask], noise[~edge_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2]}, step=self.iters)
        return
    

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return


class CondSurfPosTrainer(SurfPosTrainer):
    """ Surface Position Trainer (3D bbox) with point cloud condition """
    def __init__(self, args, train_dataset, val_dataset,):
        super(SurfPosTrainer, self).__init__()
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.use_precomputed_cond = getattr(args, 'use_precomputed_cond', False)
        
        # DDP setup
        local_rank = int(os.environ['LOCAL_RANK'])
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        # Initialize network
        model = CondSurfPosNet(self.use_cf)
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if args.weight is not None:
            self.model.module.load_state_dict(torch.load(args.weight, map_location='cpu'))
            print(f"Loaded weights from {args.weight}")

        # Initialize VAE Manager
        self.shape_vae = ShapeVAE(args, self.device)

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=1e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        if self.local_rank == 0:
            wandb.init(project='BrepGen', dir=self.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=self.train_sampler,
                                                batch_size=args.batch_size,
                                                num_workers=32)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=args.batch_size,
                                             num_workers=32)
        return

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        self.train_sampler.set_epoch(self.epoch)

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                if self.use_precomputed_cond:
                    if self.use_cf:
                        surfPos, class_label, condition = [d.to(self.device) for d in data]
                    else:
                        surfPos, condition = [d.to(self.device) for d in data]
                        class_label = None
                else:
                    if self.use_cf:
                        data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                        surfPos, class_label, point_cloud = data_cuda
                    else:
                        surfPos, point_cloud = [d.to(self.device) for d in data]
                        class_label = None
                    
                    with torch.no_grad():
                        # Get latent embedding from VAE manager
                        condition = self.shape_vae.get_embedding(point_cloud)

                bsz = len(surfPos)
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(surfPos_diffused, timesteps, class_label, condition, True)
              
                # Compute loss
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 5 == 0:
                if self.local_rank == 0:
                    # Get learning rate
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    # Calculate gradient norm
                    grad_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** (1. / 2)
                    
                    # Log additional metrics
                    log_dict = {
                        "Loss-noise": total_loss,
                        "Learning Rate": lr,
                        "Gradient Norm": grad_norm,
                    }
                    wandb.log(log_dict, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        return

    def save_model(self):
        if self.local_rank == 0:
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'surfpos_epoch_'+str(self.epoch)+'.pt'))
        return


class CondSurfZTrainer(SurfZTrainer):
    """ Surface Latent Geometry Trainer with point cloud condition """
    def __init__(self, args, train_dataset, val_dataset):
        super(SurfZTrainer, self).__init__()
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        
        # DDP setup
        local_rank = int(os.environ['LOCAL_RANK'])
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        # Initialize network
        model = CondSurfZNet(self.use_cf)
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if args.weight is not None:
            self.model.module.load_state_dict(torch.load(args.weight, map_location='cpu'))
            print(f"Loaded weights from {args.weight}")

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae, map_location='cpu'), strict=False)
        self.surf_vae = surf_vae.to(self.device).eval()

        # Load VecSetX VAE Encoder
        self.vae_encoder = vecset_ae.__dict__['point_vec1024x32_dim1024_depth24_nb'](pc_size=8192)
        checkpoint = torch.load(args.vecset_vae_weights, map_location='cpu')
        model_state_dict = checkpoint['model']
        if any(key.startswith('module.') for key in model_state_dict):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        self.vae_encoder.load_state_dict(model_state_dict)
        self.vae_encoder = self.vae_encoder.to(self.device).eval()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        if self.local_rank == 0:
            wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=self.train_sampler,
                                                batch_size=args.batch_size,
                                                num_workers=32)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=args.batch_size,
                                             num_workers=32)
        return

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        self.train_sampler.set_epoch(self.epoch)

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                if self.use_cf:
                    surfPos, surfPnt, surf_mask, class_label, point_cloud = [d.to(self.device) for d in data]
                else:
                    surfPos, surfPnt, surf_mask, point_cloud = [d.to(self.device) for d in data]
                    class_label = None

                bsz = len(surfPos)

                with torch.no_grad():
                    latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                    condition = self.vae_encoder.learn(latent_embedding)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                conditions = [surfPos]
                aug_data = []
                for d in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(d.shape).to(self.device)  
                    aug_data.append(self.noise_scheduler.add_noise(d, aug_noise, aug_timesteps))
                surfPos = aug_data[0]

                # Pass through surface VAE to sample latent z 
                with torch.no_grad():
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)     

                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ.shape).to(self.device)  
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                # Predict noise
                surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, class_label, condition, True)

                # Loss
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])        
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0 and self.local_rank == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.set_postfix(loss=total_loss.item())
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*5

        for data in self.val_dataloader:
            if self.use_cf:
                surfPos, surfPnt, surf_mask, class_label, point_cloud = [d.to(self.device) for d in data]
            else:
                surfPos, surfPnt, surf_mask, point_cloud = [d.to(self.device) for d in data]
                class_label = None
            bsz = len(surfPos)

            with torch.no_grad():
                latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                condition = self.vae_encoder.learn(latent_embedding)

            # Pass through surface VAE to sample latent z 
            with torch.no_grad():
                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z = self.surf_vae(surf_uv)
            surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)    
            tokens = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(tokens.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(tokens, noise, timesteps)
                with torch.no_grad():
                    pred = self.model(diffused, timesteps, surfPos, surf_mask, class_label, condition)
                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        if self.local_rank == 0:
            mse = [loss/total_count for loss in total_loss]
            self.model.train() # set to train
            wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        return

    def save_model(self):
        if self.local_rank == 0:
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'surfz_epoch_'+str(self.epoch)+'.pt'))
        return

