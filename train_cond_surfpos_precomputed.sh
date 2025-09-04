#!/bin/bash

# Training script for CondSurfPosTrainer with precomputed embeddings
python train.py \
    --model cond_surfpos \
    --data_path /path/to/your/data \
    --save_dir proj_log/deepcad_cond_surfpos_precomputed \
    --batch_size 32 \
    --env deepcad_cond_surfpos_precomputed \
    --vecset_vae_weights /home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/checkpoint-110.pth \
    --precomputed_emd \
    --cf