import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import trimesh
from network import *
from network_conditional import CondSurfPosNetFM, CondSurfZNetFM
from utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    generate_random_string,
    construct_brep,
    detect_shared_vertex,
    detect_shared_edge,
    joint_optimize,
    save_points_and_lines_as_ply,
    bbox_corners,
)
from dataset import normalize_point_cloud, resample_point_cloud
from VecSetX.vecset.models import autoencoder as vecset_ae


text2int = {'uncond':0,
            'bathtub':1,
            'bed':2, 
            'bench':3, 
            'bookshelf':4,
            'cabinet':5, 
            'chair':6, 
            'couch':7, 
            'lamp':8, 
            'sofa':9, 
            'table':10
            }


def load_point_cloud(ply_path, num_points=8192):
    """Load and preprocess point cloud from .ply file"""
    # Load point cloud
    mesh = trimesh.load(ply_path)
    point_cloud = mesh.vertices
    
    # Normalize the point cloud
    point_cloud = normalize_point_cloud(point_cloud)
    
    # Resample to the required size
    point_cloud = resample_point_cloud(point_cloud, num_points)
        
    return torch.FloatTensor(point_cloud)


def sample(eval_args, point_cloud_path=None):

    # Inference configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces']
    num_edges = eval_args['num_edges']
    num_inference_steps = eval_args.get('num_inference_steps', 100)

    if eval_args['use_cf']:
        class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                       [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    surfPos_model = CondSurfPosNetFM(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight']))
    surfPos_model = surfPos_model.to(device).eval()

    # Load VecSetX VAE Encoder
    vae_encoder = vecset_ae.__dict__['point_vec1024x32_dim1024_depth24_nb'](pc_size=8192)
    vae_weights_path = eval_args.get('vecset_vae_weights', '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/checkpoint-110.pth')
    checkpoint = torch.load(vae_weights_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in model_state_dict):
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    vae_encoder.load_state_dict(model_state_dict)
    vae_encoder = vae_encoder.to(device).eval()

    surfZ_model = CondSurfZNetFM(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight']))
    surfZ_model = surfZ_model.to(device).eval()

    surf_vae = AutoencoderKLFastDecode(in_channels=3,
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
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight']), strict=False)
    surf_vae = surf_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
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
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight']), strict=False)
    edge_vae = edge_vae.to(device).eval()


    with torch.no_grad():
        with torch.cuda.amp.autocast():
        
            # Load or generate point cloud as condition
            if point_cloud_path and os.path.exists(point_cloud_path):
                print(f"Loading point cloud from {point_cloud_path}")
                point_cloud = load_point_cloud(point_cloud_path).unsqueeze(0).to(device)
                # Repeat for batch size
                point_cloud = point_cloud.repeat(batch_size, 1, 1)
            else:
                print("Generating random point cloud as condition")
                # Generate random point cloud as condition (8192 points as in the training)
                point_cloud = torch.randn(batch_size, 8192, 3).to(device)
            
            # Encode point cloud using VAE encoder
            with torch.no_grad():
                latent_embedding = vae_encoder.encode(point_cloud)['x']
                condition = vae_encoder.learn(latent_embedding)
        
            ###########################################
            # STEP 1-1: generate the surface position #
            ###########################################
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device) # z_0

            timesteps = torch.linspace(0, 1, num_inference_steps, device=device)
            dt = timesteps[1] - timesteps[0]

            for i in tqdm(range(len(timesteps) - 1)):
                t_cur, t_next = timesteps[i], timesteps[i+1]
                dt = t_next - t_cur

                # Predictor step (Euler)
                time_input_cur = t_cur.expand(batch_size) * 999.0
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred_cur = surfPos_model(_surfPos_, time_input_cur.repeat(2), class_label, _condition_)
                    pred_cur = pred_cur[:batch_size] * (1+w) - pred_cur[batch_size:] * w
                else:
                    pred_cur = surfPos_model(surfPos, time_input_cur, class_label, condition)
                
                surfPos_next_hat = surfPos + pred_cur * dt

                # Corrector step
                time_input_next = t_next.expand(batch_size) * 999.0
                if class_label is not None:
                    _surfPos_next_hat_ = surfPos_next_hat.repeat(2,1,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred_next = surfPos_model(_surfPos_next_hat_, time_input_next.repeat(2), class_label, _condition_)
                    pred_next = pred_next[:batch_size] * (1+w) - pred_next[batch_size:] * w
                else:
                    pred_next = surfPos_model(surfPos_next_hat, time_input_next, class_label, condition)
                
                # Update surfPos with the average of the two derivatives
                surfPos = surfPos + (pred_cur + pred_next) * dt / 2
           
            #######################################
            # STEP 1-2: remove duplicate surfaces #
            #######################################
            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum()>=1 or same_rev.sum()>=1:
                        continue # repeat value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
                bboxes = non_repeat.reshape(len(non_repeat),-1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces-len(bboxes),6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces-len(bboxes))==0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)

            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()


            #################################
            # STEP 1-3:  generate surface z #
            #################################
            surfZ = randn_tensor((batch_size, num_surfaces, 48)).to(device)
            
            for i in tqdm(range(len(timesteps) - 1)):
                t_cur, t_next = timesteps[i], timesteps[i+1]
                dt = t_next - t_cur

                # Predictor step (Euler)
                time_input_cur = t_cur.expand(batch_size) * 999.0
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred_cur = surfZ_model(_surfZ_, time_input_cur.repeat(2), _surfPos_, _surfMask_, class_label, _condition_)
                    pred_cur = pred_cur[:batch_size] * (1+w) - pred_cur[batch_size:] * w
                else:
                    pred_cur = surfZ_model(surfZ, time_input_cur, surfPos, surfMask, class_label, condition)
                
                surfZ_next_hat = surfZ + pred_cur * dt

                # Corrector step
                time_input_next = t_next.expand(batch_size) * 999.0
                if class_label is not None:
                    _surfZ_next_hat_ = surfZ_next_hat.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred_next = surfZ_model(_surfZ_next_hat_, time_input_next.repeat(2), _surfPos_, _surfMask_, class_label, _condition_)
                    pred_next = pred_next[:batch_size] * (1+w) - pred_next[batch_size:] * w
                else:
                    pred_next = surfZ_model(surfZ_next_hat, time_input_next, surfPos, surfMask, class_label, condition)

                # Update surfZ with the average of the two derivatives
                surfZ = surfZ + (pred_cur + pred_next) * dt / 2

    for batch_idx in range(batch_size):
        random_string = generate_random_string(15)
        savename = f'sample_{batch_idx}'

        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad].detach().cpu().numpy()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                surf_ncs_cad = surf_vae(torch.FloatTensor(surf_z_cad).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()

        surf_wcs_cad = []
        for ncs, pos in zip(surf_ncs_cad, surf_pos_cad):
            bcenter, bsize = compute_bbox_center_and_size(pos[0:3], pos[3:])
            wcs = ncs * (bsize / 2) + bcenter
            surf_wcs_cad.append(wcs)
        surf_wcs_cad = np.array(surf_wcs_cad)
        save_points_and_lines_as_ply(surf_wcs_cad, os.path.join(save_folder, f'{savename}'), bboxes=surf_pos_cad, condition= point_cloud[batch_idx].cpu().numpy() if point_cloud_path else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc',
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--point_cloud", type=str, default=None,
                        help="Path to the point cloud .ply file for conditional generation")
    args = parser.parse_args()

    # Load evaluation config
    with open('eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    eval_args = config[args.mode]

    sample(eval_args, args.point_cloud)