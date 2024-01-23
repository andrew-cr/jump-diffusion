
# Repository

This repository is built on top the open source repo https://github.com/NVlabs/edm

# Dataset

Download the RoboDesk dataset from [this repository](git@github.com:s-tian/vp2.git) into `datasets/`. It should be structured:
```
datasets/
    robodesk/
        robodesk_flat_block_off_table/
            noise_0.1_256.hdf5
        robodesk_open_slide/
            noise_0.1_256.hdf5
        ...
```

# Training


## Score network (run on node with 4 GPUs)
```
torchrun --standalone --nproc_per_node=4 train.py --path=datasets/robodesk/ --data_class RoboDeskVideoDataset --task_distribution even-35-1 --resolution 32 --batch 4 --seed 1 --exist 1,1,1 --observed 0,1,1 --tick 1 --sample 50 --cache False --ema 0.005 --arch=fdm --lr=1e-4 --augment=0 --workers 5 --latent_space False --jump_diffusion True --pred_x0 0,0,0,1,1 --detach_jump_grads True --any_dimension_deletion True --highly_nonisotropic True --only_use_neighbours 2 --duplicate_videos_in_batch 8
```

## Jump rate and index pred network (run on node with 4 GPUs)
```
torchrun --standalone --nproc_per_node=4 train.py --path=datasets/robodesk/ --data_class RoboDeskVideoDataset --task_distribution even-35-1 --resolution 32 --batch 4 --seed 1 --exist 1,1,1 --observed 0,1,1 --tick 1 --sample 50 --cache False --ema 0.005 --arch=just_jump --lr=1e-4 --augment=0 --workers 6 --latent_space False --jump_diffusion True --pred_x0 0,0,0,1,1 --detach_jump_grads True --any_dimension_deletion True --highly_nonisotropic True --duplicate_videos_in_batch 4 --jump_per_index True --jump_net_embedder_type unet --just_jump_loss True
```

# Sampling
We provide pretrained checkpoints [here](https://drive.google.com/drive/folders/1OzrH2Tg7q4SwwfcbYsPoLLvHW8_Ov4OT?usp=sharing) ~~[here](https://drive.google.com/drive/folders/1O2LXpEGk0GGFUnfyWEPXcCsO4dITThSf?usp=sharing)~~. You can download them to `models/` and sample with them using the following command: 

```
torchrun --standalone --nproc_per_node=1 generate.py --steps 100 --S_churn 50 --S_noise 1.007 --seeds=0 --batch 1 --cond_endpoints --network=models/score/network-snapshot-001100.pkl --jump_network=models/jump_rate_and_index/network-snapshot-000450.pkl
```

# Environment

```
pytorch
rdkit
scipy
click
wandb
tqdm
moviepy
Pillow
einops
```