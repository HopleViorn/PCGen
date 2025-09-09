import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import CondSurfPosData, text2int
from vae_models import ShapeVAE
from utils import rotate_point_cloud

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser('Pre-calculate conditions with DDP')
    
    # --- Data arguments ---
    parser.add_argument('--input_data', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data', help='path to input data')
    parser.add_argument('--input_list', type=str, default=None, help='path to data split list')
    parser.add_argument('--output_dir', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/conditions', help='path to save pre-calculated conditions')
    parser.add_argument('--max_face', type=int, default=30, help='maximum number of faces per shape')
    parser.add_argument('--max_edge', type=int, default=20, help='maximum number of edges per face')
    parser.add_argument('--bbox_scaled', type=int, default=3, help='scaled value for bbox')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold for filtering duplicates')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for preprocessing per GPU')
    
    # --- VAE arguments ---
    parser.add_argument('--vae_weight', type=str, default='proj_log/vae_deepcad/surf_epoch_1000.pt', help='path to vae weights')
    parser.add_argument('--vae_encoder_type', type=str, default='hy3dshape', help='type of vae encoder')
    parser.add_argument('--cond_dim', type=int, default=256, help='dimension of condition')
    parser.add_argument('--z_dim', type=int, default=256, help='dimension of latent')
    parser.add_argument('--vae_hidden_dim', type=int, default=256, help='hidden dimension of vae')
    parser.add_argument('--vae_num_layers', type=int, default=4, help='number of layers in vae')
    parser.add_argument('--vae_num_heads', type=int, default=4, help='number of heads in vae')
    args = parser.parse_args()

    # DDP setup
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize VAE Manager and wrap with DDP
    shape_vae = ShapeVAE(args, device).to(device)
    shape_vae = DDP(shape_vae, device_ids=[local_rank])

    # Initialize Dataset and Dataloader for batch processing
    dataset = CondSurfPosData(args.input_data, args.input_list, validate=False, aug=False, args=args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=sampler)
    
    if local_rank == 0:
        # --- Debugging: Check if the specific item is in the dataset ---
        found_item = False
        for item_path in dataset.data:
            path_to_check = item_path[0] if isinstance(item_path, tuple) else item_path
            if '00522503' in path_to_check:
                print(f"\n--- !!! Found '00522503' in dataset.data: {path_to_check} !!! ---\n")
                found_item = True
                break
        if not found_item:
            print("\n--- !!! Did not find '00522503' in the initial dataset.data list. !!! ---\n")
        # --- End Debugging ---

        print(f"Processing {len(dataset)} data points on {world_size} GPUs in batches of {args.batch_size}...")

    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        surf_pos_batch, point_cloud_batch = batch_data
        
        unprocessed_indices = []
        unprocessed_data_paths = []
        start_idx = batch_idx * args.batch_size * world_size + local_rank * args.batch_size

        for i in range(point_cloud_batch.shape[0]):
            item_idx = start_idx + i
            if item_idx >= len(dataset.data): continue

            data_path = dataset.data[item_idx]
            if isinstance(data_path, tuple):
                data_path, _ = data_path

            filename_pkl = os.path.basename(data_path)
            filename_pkl_no_ext = os.path.splitext(filename_pkl)[0]
            parts = filename_pkl_no_ext.split('_')
            subdir = parts[0]
            filename_base = '_'.join(parts[1:])
            output_dir_per_item = os.path.join(args.output_dir, subdir, filename_base)

            if not os.path.exists(output_dir_per_item):
                unprocessed_indices.append(i)
                unprocessed_data_paths.append(data_path)

        if not unprocessed_indices:
            continue

        point_cloud_batch = point_cloud_batch[unprocessed_indices]
        point_cloud_batch = point_cloud_batch.to(device).half()

        with torch.no_grad():
            latent_batch = shape_vae.module.get_latent(point_cloud_batch)
            condition_batch = shape_vae.module.decode(latent_batch)
        
        for i in range(condition_batch.shape[0]):
            data_path = unprocessed_data_paths[i]
            filename_pkl = os.path.basename(data_path)
            filename_pkl_no_ext = os.path.splitext(filename_pkl)[0]
            parts = filename_pkl_no_ext.split('_')
            subdir = parts[0]
            filename_base = '_'.join(parts[1:])
            output_dir_per_item = os.path.join(args.output_dir, subdir, filename_base)
            os.makedirs(output_dir_per_item, exist_ok=True)

            save_path_original = os.path.join(output_dir_per_item, 'no_rot.pkl')
            with open(save_path_original, 'wb') as f:
                pickle.dump(condition_batch[i].cpu().numpy(), f)

            save_path_latent = os.path.join(output_dir_per_item, 'no_rot_latent.pkl')
            with open(save_path_latent, 'wb') as f:
                pickle.dump(latent_batch[i].cpu().numpy(), f)

        pc_numpy_batch = point_cloud_batch.cpu().numpy()
        for axis in ['x', 'y', 'z']:
            for angle in [90, 180, 270]:
                rotated_pc_list = []
                for pc_numpy in pc_numpy_batch:
                    if args.vae_encoder_type == 'hy3dshape':
                        points, normals, sharp_edges = pc_numpy[:, :3], pc_numpy[:, 3:6], pc_numpy[:, 6:]
                        rotated_points = rotate_point_cloud(points, angle, axis)
                        rotated_normals = rotate_point_cloud(normals, angle, axis)
                        rotated_pc_numpy = np.concatenate([rotated_points, rotated_normals, sharp_edges], axis=1)
                    else:
                        rotated_pc_numpy = rotate_point_cloud(pc_numpy, angle, axis)
                    rotated_pc_list.append(rotated_pc_numpy)
                
                point_cloud_rotated_batch = torch.FloatTensor(np.stack(rotated_pc_list)).to(device).half()

                with torch.no_grad():
                    latent_rotated_batch = shape_vae.module.get_latent(point_cloud_rotated_batch)
                    condition_rotated_batch = shape_vae.module.decode(latent_rotated_batch)

                for i in range(condition_rotated_batch.shape[0]):
                    data_path = unprocessed_data_paths[i]
                    filename_pkl = os.path.basename(data_path)
                    filename_pkl_no_ext = os.path.splitext(filename_pkl)[0]
                    parts = filename_pkl_no_ext.split('_')
                    subdir = parts[0]
                    filename_base = '_'.join(parts[1:])
                    output_dir_per_item = os.path.join(args.output_dir, subdir, filename_base)

                    save_filename = f"rot_{axis}_{angle}.pkl"
                    save_path_rotated = os.path.join(output_dir_per_item, save_filename)
                    with open(save_path_rotated, 'wb') as f:
                        pickle.dump(condition_rotated_batch[i].cpu().numpy(), f)

                    save_filename_latent = f"rot_{axis}_{angle}_latent.pkl"
                    save_path_latent_rotated = os.path.join(output_dir_per_item, save_filename_latent)
                    with open(save_path_latent_rotated, 'wb') as f:
                        pickle.dump(latent_rotated_batch[i].cpu().numpy(), f)

    cleanup()
    if local_rank == 0:
        print("Finished pre-calculating all conditions.")


if __name__ == '__main__':
    main()