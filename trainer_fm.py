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
from network_conditional import CondSurfPosNetFM, CondSurfZNetFM
from VecSetX.vecset.models import autoencoder as vecset_ae
from trainer import CondSurfPosTrainer, CondSurfZTrainer
from transport import Transport

class CondSurfPosTrainerFM(CondSurfPosTrainer):
    """ Surface Position Trainer (3D bbox) with point cloud condition """
    def __init__(self, args, train_dataset, val_dataset,):
        super(CondSurfPosTrainer, self).__init__( args, train_dataset, val_dataset)
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        
        # DDP setup
        local_rank = int(os.environ['LOCAL_RANK'])
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        # Initialize network
        model = CondSurfPosNetFM(self.use_cf)
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if args.weight is not None:
            self.model.module.load_state_dict(torch.load(args.weight, map_location='cpu'))
            print(f"Loaded weights from {args.weight}")

        # Load VecSetX VAE Encoder
        self.vae_encoder = vecset_ae.__dict__['point_vec1024x32_dim1024_depth24_nb'](pc_size=8192)
        checkpoint = torch.load(args.vecset_vae_weights, map_location='cpu')
        model_state_dict = checkpoint['model']
        if any(key.startswith('module.') for key in model_state_dict):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        self.vae_encoder.load_state_dict(model_state_dict)
        self.vae_encoder = self.vae_encoder.to(self.device).eval()

        self.loss_fn = nn.MSELoss()
        self.transport = Transport(path_type='LINEAR', prediction='VELOCITY')

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
                if self.use_cf:
                    data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                    surfPos, class_label, point_cloud = data_cuda
                else:
                    surfPos, point_cloud = [d.to(self.device) for d in data]
                    class_label = None
                
                with torch.no_grad():
                    latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                    condition = self.vae_encoder.learn(latent_embedding)

                self.optimizer.zero_grad()

                # Use the new transport module to compute the loss
                model_kwargs = {'class_label': class_label, 'condition': condition, 'is_train': True}
                loss_terms = self.transport.training_losses(self.model.forward, surfPos, model_kwargs)
                total_loss = loss_terms['loss']
             
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
                        "Loss-fm": total_loss,
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
                data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                surfPos, class_label, point_cloud = data_cuda
            else:
                surfPos, point_cloud = [d.to(self.device) for d in data]
                class_label = None
            bsz = len(surfPos)

            with torch.no_grad():
                latent_embedding = self.vae_encoder.encode(point_cloud)['x']
                condition = self.vae_encoder.learn(latent_embedding)

            total_count += len(surfPos)
            
            for idx, step in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
                # Evaluate at timestep
                t = torch.full((bsz,), step, device=self.device)
                z_0 = torch.randn_like(surfPos)
                z_t = (1 - t.view(-1, 1, 1)) * z_0 + t.view(-1, 1, 1) * surfPos
                target_vector = surfPos - z_0
                with torch.no_grad():
                    # Scale t from [0, 1] to [0, 999] for time embedding
                    time_input = t * 999.0
                    pred = self.model(z_t, time_input, class_label, condition)
                loss = mse_loss(pred, target_vector).mean((1,2)).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        if self.local_rank == 0:
            mse = [loss/total_count for loss in total_loss]
            self.model.train() # set to train
            wandb.log({"Val-0.1": mse[0], "Val-0.3": mse[1], "Val-0.5": mse[2], "Val-0.7": mse[3], "Val-0.9": mse[4]}, step=self.iters)
        return


class CondSurfZTrainerFM(CondSurfZTrainer):
    """ Surface Latent Geometry Trainer with point cloud condition """
    def __init__(self, args, train_dataset, val_dataset):
        super(CondSurfZTrainer, self).__init__( args, train_dataset, val_dataset)
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
        model = CondSurfZNetFM(self.use_cf)
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
        self.transport = Transport(path_type='LINEAR', prediction='VELOCITY')

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

                # Pass through surface VAE to sample latent z
                with torch.no_grad():
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)

                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                
                self.optimizer.zero_grad()

                # Use the new transport module to compute the loss
                model_kwargs = {'surfPos': surfPos, 'surf_mask': surf_mask, 'class_label': class_label, 'condition': condition, 'is_train': True}
                
                # Use a lambda function for a cleaner model forward pass
                loss_terms = self.transport.training_losses(
                    lambda surfZ, time_input, **kwargs: self.model(surfZ, time_input, **kwargs),
                    surfZ,
                    model_kwargs
                )
                
                # The loss needs to be masked
                pred = loss_terms['pred']
                target = loss_terms['target']
                total_loss = self.loss_fn(pred[~surf_mask], target[~surf_mask])
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0 and self.local_rank == 0:
                wandb.log({"Loss-fm": total_loss}, step=self.iters)

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
            
            for idx, step in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
                # Evaluate at timestep 
                t = torch.full((bsz,), step, device=self.device)
                z_0 = torch.randn_like(tokens)
                z_t = (1 - t.view(-1, 1, 1)) * z_0 + t.view(-1, 1, 1) * tokens
                target_vector = tokens - z_0
                with torch.no_grad():
                    time_input = t * 999.0
                    pred = self.model(z_t, time_input, surfPos, surf_mask, class_label, condition)
                loss = mse_loss(pred[~surf_mask], target_vector[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        if self.local_rank == 0:
            mse = [loss/total_count for loss in total_loss]
            self.model.train() # set to train
            wandb.log({"Val-0.1": mse[0], "Val-0.3": mse[1], "Val-0.5": mse[2], "Val-0.7": mse[3], "Val-0.9": mse[4]}, step=self.iters)
        return

    def save_model(self):
        if self.local_rank == 0:
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'surfz_epoch_'+str(self.epoch)+'.pt'))
        return