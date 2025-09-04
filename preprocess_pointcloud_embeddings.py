import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import trimesh
import argparse
from torch.utils.data import DataLoader, Dataset
from VecSetX.vecset.models import autoencoder as vecset_ae
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class PointCloudDataset(Dataset):
    """Dataset class for point clouds"""
    def __init__(self, ply_files, max_points=4096):
        self.ply_files = ply_files
        self.max_points = max_points
        
    def __len__(self):
        return len(self.ply_files)
    
    def __getitem__(self, idx):
        ply_path = self.ply_files[idx]
        try:
            # Load point cloud
            point_cloud = trimesh.load(ply_path).vertices
            
            # Randomly sample points if needed
            if point_cloud.shape[0] > self.max_points:
                indices = np.random.choice(point_cloud.shape[0], self.max_points, replace=False)
                point_cloud = point_cloud[indices]
            elif point_cloud.shape[0] < self.max_points:
                # Pad with zeros if fewer points
                padding = np.zeros((self.max_points - point_cloud.shape[0], 3))
                point_cloud = np.vstack([point_cloud, padding])
            
            return torch.FloatTensor(point_cloud), ply_path
        except Exception as e:
            print(f"Error loading {ply_path}: {e}")
            # Return zeros if there's an error
            return torch.zeros(self.max_points, 3), ply_path

def load_vae_encoder(weights_path, model_name='point_vec1024x32_dim1024_depth24_nb'):
    """Load the VecSetX VAE encoder"""
    print(f"Loading VAE encoder from {weights_path}")
    vae_encoder = vecset_ae.__dict__[model_name](pc_size=4096)
    
    # Load pretrained weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in model_state_dict):
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    vae_encoder.load_state_dict(model_state_dict)
    vae_encoder = vae_encoder.cuda()
    
    return vae_encoder

def compute_and_save_embeddings_batch(input_dir, output_dir, vae_encoder, batch_size=128, max_points=4096):
    """Compute embeddings for all point clouds in batches and save them"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .ply files
    ply_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    print(f"Found {len(ply_files)} point cloud files")
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(ply_files, max_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    
    # Process in batches
    with torch.no_grad():
        for point_clouds, paths in tqdm(dataloader, desc="Processing point clouds"):
            try:
                # Move to GPU
                point_clouds = point_clouds.cuda()
                
                # Compute embeddings
                latent_embeddings = vae_encoder.module.encode(point_clouds)['x']
                conditions = vae_encoder.module.learn(latent_embeddings)
                
                # Save embeddings
                for i, path in enumerate(paths):
                    # Determine output path
                    rel_path = os.path.relpath(path, input_dir)
                    output_path = os.path.join(output_dir, rel_path.replace('.ply', '.pkl'))
                    
                    # Create output directory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save embedding
                    with open(output_path, 'wb') as f:
                        pickle.dump(conditions[i].cpu().numpy(), f)
                        
            except Exception as e:
                print(f"Error processing batch: {e}")

def setup_ddp(rank, world_size):
    """Setup DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleanup DDP environment"""
    dist.destroy_process_group()

def compute_and_save_embeddings_ddp(rank, world_size, input_dir, output_dir, vae_weights, batch_size=256):
    """Compute embeddings using DDP"""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Load VAE encoder
    vae_encoder = load_vae_encoder(vae_weights)
    vae_encoder = DDP(vae_encoder, device_ids=[rank])
    
    # Get all .ply files
    ply_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    
    # Distribute files across GPUs
    files_per_gpu = len(ply_files) // world_size
    start_idx = rank * files_per_gpu
    end_idx = start_idx + files_per_gpu if rank < world_size - 1 else len(ply_files)
    gpu_files = ply_files[start_idx:end_idx]
    
    print(f"GPU {rank}: Processing {len(gpu_files)} files")
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(gpu_files, 4096)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32, sampler=sampler)
    
    # Process in batches
    with torch.no_grad():
        for point_clouds, paths in tqdm(dataloader, desc=f"GPU {rank} processing"):
            try:
                # Move to GPU
                point_clouds = point_clouds.cuda(non_blocking=True)
                
                # Compute embeddings
                latent_embeddings = vae_encoder.module.encode(point_clouds)['x']
                conditions = vae_encoder.module.learn(latent_embeddings)
                
                # Save embeddings
                for i, path in enumerate(paths):
                    # Determine output path
                    rel_path = os.path.relpath(path, input_dir)
                    output_path = os.path.join(output_dir, rel_path.replace('.ply', '.pkl'))
                    
                    # Create output directory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save embedding
                    with open(output_path, 'wb') as f:
                        pickle.dump(conditions[i].cpu().numpy(), f)
                        
            except Exception as e:
                print(f"GPU {rank} error processing batch: {e}")
    
    # Cleanup
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description='Precompute point cloud embeddings')
    parser.add_argument('--input_dir', type=str, 
                        default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad',
                        help='Input directory containing point cloud files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_emd',
                        help='Output directory for embeddings')
    parser.add_argument('--vae_weights', type=str,
                        default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/checkpoint-110.pth',
                        help='Path to VecSetX VAE weights')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--use_ddp', action='store_true',
                        help='Use DDP for multi-GPU processing')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.use_ddp:
        # Use DDP for multi-GPU processing
        mp.spawn(compute_and_save_embeddings_ddp, 
                 args=(args.gpus, args.input_dir, args.output_dir, args.vae_weights, args.batch_size),
                 nprocs=args.gpus,
                 join=True)
    else:
        # Load VAE encoder
        print("Loading VAE encoder...")
        vae_encoder = load_vae_encoder(args.vae_weights)
        vae_encoder = torch.nn.DataParallel(vae_encoder)
        vae_encoder = vae_encoder.cuda().eval()
        
        # Compute and save embeddings
        print("Computing and saving embeddings...")
        compute_and_save_embeddings_batch(args.input_dir, args.output_dir, vae_encoder, args.batch_size)
    
    print("Done!")

if __name__ == "__main__":
    main()