import os
import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add paths for hy3dshape and BrepGen modules



sys.path.insert(0, './')
sys.path.insert(0, './hy3dshape')

from vae_models import ShapeVAE

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


from network import AutoencoderKLFastDecode, SurfZNet
from network_conditional import CondSurfPosNet
from diffusers import DDPMScheduler, PNDMScheduler
from utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    generate_random_string,
    save_points_and_lines_as_ply,
)


try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


def sample_from_image(eval_args, image_path):
    # --- Part 1: Image to Latent Code ---
    print("Step 1: Generating latent code from the input image...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Hunyuan3D pipeline
    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device=device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)

    # Generate latent code (condition) from the image
    # By setting output_type='latent', the pipeline returns the latent tensor instead of a mesh
    with torch.no_grad():
        # The pipeline returns a list of latents, one for each image in the batch.
        # Since we have one image, we take the first element.
        latents = pipeline_shapegen(image=image)
        vae = ShapeVAE(argparse.Namespace(**{'vae_encoder_type':'hy3dshape'}), device = device)
        # condition = vae.decode(latents)
        condition = latents

    print("Latent code generated successfully.")

    # --- Part 2: Latent Code to 3D Surface ---
    print("\nStep 2: Sampling 3D surface using the generated latent code...")
    batch_size = eval_args['batch_size']
    bbox_threshold = eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces']
    condition = condition.repeat(batch_size, 1, 1)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load models
    surfPos_model = CondSurfPosNet(use_cf=False)
    # surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight']))
    surfPos_model.load_state_dict(torch.load('/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/proj_log/deepcad_ldm_surfpos_hy_latent/surfpos_epoch_15.pt'))
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(use_cf=False)
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight']))
    surfZ_model = surfZ_model.to(device).eval()

    surf_vae = AutoencoderKLFastDecode(
        in_channels=3, out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512], layers_per_block=2, act_fn='silu',
        latent_channels=3, norm_num_groups=32, sample_size=512
    )
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight']), strict=False)
    surf_vae = surf_vae.to(device).eval()

    # Initialize schedulers
    pndm_scheduler = PNDMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon')
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon', clip_sample=True, clip_sample_range=3)


    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # STEP 2-1: Generate surface positions
            print("Generating surface positions...")
            surfPos = randn_tensor((batch_size, num_surfaces, 6), device=device)
            pndm_scheduler.set_timesteps(200)
            for t in tqdm(pndm_scheduler.timesteps[:158]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps, None, condition)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample

            surfPos = surfPos.repeat(1, 2, 1)
            num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps, None, condition)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample

            # STEP 2-2: Remove duplicate surfaces
            print("Deduplicating surfaces...")
            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1, torch.Size([2, 3])).detach().cpu().numpy(), 4)
                non_repeat = bboxes[:1]
                for bbox in bboxes[1:]:
                    diff = np.max(np.abs(non_repeat - bbox), axis=(-1, -2))
                    if not np.any(diff < bbox_threshold):
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
                bboxes = non_repeat.reshape(len(non_repeat), -1)

                surf_mask = torch.zeros((1, len(bboxes)), device=device) == 1
                bbox_padded = torch.cat([torch.FloatTensor(bboxes).to(device), torch.zeros(num_surfaces - len(bboxes), 6, device=device)])
                mask_padded = torch.cat([surf_mask, torch.ones(1, num_surfaces - len(bboxes), device=device)==1], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)

            surfPos = torch.stack(surfPos_deduplicate)
            surfMask = torch.vstack(surfMask_deduplicate)

            # STEP 2-3: Generate surface geometry (z)
            print("Generating surface geometry...")
            surfZ = randn_tensor((batch_size, num_surfaces, 48), device=device)
            pndm_scheduler.set_timesteps(200)
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()
                # Note: The original surfZ_model in sample_cond_surface.py does not take condition.
                # We assume the CondSurfZNet should be used here. If not, this part needs adjustment.
                # For now, we pass `condition` to surfZ_model, assuming it can handle it.
                pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label=None) #, condition=condition)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample

    # --- Part 3: Decode and Save Results ---
    print("\nStep 3: Decoding surfaces and saving results...")
    for batch_idx in range(batch_size):
        savename = f'sample_from_image_{batch_idx}'

        surfMask_cad = surfMask[batch_idx].detach().cpu()
        surf_z_cad = surfZ[batch_idx][~surfMask_cad].detach().cpu()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad].detach().cpu().numpy()

        if surf_z_cad.shape[0] == 0:
            print(f"Warning: No surfaces generated for sample {batch_idx}. Skipping.")
            continue

        with torch.no_grad(), torch.cuda.amp.autocast():
            surf_ncs_cad = surf_vae(surf_z_cad.cuda().unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4])))
            surf_ncs_cad = surf_ncs_cad.permute(0, 2, 3, 1).detach().cpu().numpy()

        surf_wcs_cad = []
        for ncs, pos in zip(surf_ncs_cad, surf_pos_cad):
            bcenter, bsize = compute_bbox_center_and_size(pos[0:3], pos[3:])
            wcs = ncs * (bsize / 2) + bcenter
            surf_wcs_cad.append(wcs)
        surf_wcs_cad = np.array(surf_wcs_cad)

        output_path = os.path.join(save_folder, savename)
        save_points_and_lines_as_ply(surf_wcs_cad, output_path, bboxes=surf_pos_cad)
        print(f"Saved result for sample {batch_idx} to {output_path}.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file for conditional generation.")
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture'], default='deepcad',
                        help="Choose evaluation mode [abc/deepcad/furniture] (default: abc)")
    args = parser.parse_args()

    # Load evaluation config
    with open('eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    eval_args = config[args.mode]

    sample_from_image(eval_args, args.image)