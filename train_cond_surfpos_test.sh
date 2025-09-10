#!/bin/bash\

### Train the Latent Diffusion Model ###
# --data_aug is optional
# max_face 30, max_edge 20 for deepcad 
# max_face 50, max_edge 30 for abc/furniture
# --surfvae refer to the surface vae weights 
# --edgevae refer to the edge vae weights 

## Training DeepCAD Latent Diffusion Model ###  
# python ldm.py --data data \
#     --option cond_surfpos \
#     --gpu 4 5 6 7 \
#     --batch_size 128 \
#     --env deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 1 \
#     --max_face 30 --max_edge 20

    # --weight /home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/proj_log/deepcad_ldm_surfpos/surfpos_epoch_199.pt \

    # --weight /home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/proj_log/deepcad_ldm_surfpos_hy/surfpos_epoch_4.pt \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=32 torchrun --nproc_per_node=2 ldm.py --data data \
    --option cond_surfpos \
    --gpu 0 1 \
    --batch_size 128 \
	--vae_encoder_type hy3dshape \
	--use_precomputed_cond \
    --data_aug \
    --env deepcad_ldm_surfpos_hy_decode --train_nepoch 3000 --test_nepoch 2000 --save_nepoch 1 \
    --max_face 30 --max_edge 20