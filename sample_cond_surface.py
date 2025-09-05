import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import trimesh
from network import *
from network_conditional import CondSurfPosNet
from diffusers import DDPMScheduler, PNDMScheduler
from OCC.Extend.DataExchange import write_stl_file, write_step_file
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

    if eval_args['use_cf']:
        class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                       [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    surfPos_model = CondSurfPosNet(eval_args['use_cf'])
    # surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight']))
    surfPos_model.load_state_dict(torch.load('proj_log/deepcad_ldm_surfpos/surfpos_epoch_183.pt'))
    surfPos_model = surfPos_model.to(device).eval()

    # Load VecSetX VAE Encoder
    vae_encoder = vecset_ae.__dict__['point_vec1024x32_dim1024_depth24_nb'](pc_size=8192)
    # NOTE: You need to provide the correct path to the pretrained VAE weights
    vae_weights_path = eval_args.get('vecset_vae_weights', '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/checkpoint-110.pth')
    checkpoint = torch.load(vae_weights_path, map_location='cpu')
    model_state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in model_state_dict):
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    vae_encoder.load_state_dict(model_state_dict)
    vae_encoder = vae_encoder.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight']))
    # surfZ_model.load_state_dict(torch.load('proj_log/deepcad_ldm_surfz_1/surfz_epoch_100.pt'))
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

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 


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
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device)

            pndm_scheduler.set_timesteps(200)
            for t in tqdm(pndm_scheduler.timesteps[:158]):#
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label, _condition_)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label, condition)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
           
            # Late increase for ABC/DeepCAD (slightly more efficient)
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1,2,1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _condition_ = condition.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label, _condition_)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label, condition)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
           

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
            
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps): 
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample

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
        # save_points_and_lines_as_ply(surf_wcs_cad, os.path.join(save_folder, f'{savename}_surf_pts.ply'))
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