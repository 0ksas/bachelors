#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 pytorch-CycleGAN-and-pix2pix/test.py --dataroot Dataset/Combined_V2 --name real2viceCycleGAN_V2 --model cycle_gan --no_dropout