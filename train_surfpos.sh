#!/bin/bash\

### Train the Latent Diffusion Model ###
# --data_aug is optional
# max_face 30, max_edge 20 for deepcad 
# max_face 50, max_edge 30 for abc/furniture
# --surfvae refer to the surface vae weights 
# --edgevae refer to the edge vae weights 

## Training DeepCAD Latent Diffusion Model ###  
python ldm.py --data data \
    --gpu 4 5 6 7 \
    --batch_size 512 \
    --env deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 10 \
    --max_face 30 --max_edge 20

# python ldm.py --data data_process/deepcad_parsed \
#     --option surfz \
#     --surfvae deepcad_vae_surf.pt --gpu 4 5 6 7\
#     --env deepcad_ldm_surfz --train_nepoch 10 --batch_size 1024 \
#     --max_face 30 --max_edge 20