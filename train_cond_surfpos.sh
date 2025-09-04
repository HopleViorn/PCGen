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

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 ldm.py --data data \
    --option cond_surfpos \
    --gpu 4 5 6 7 \
    --batch_size 256 \
    --env deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 1 \
    --max_face 30 --max_edge 20