#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 MSPC/test.py --dataroot Dataset/Combined --name real2viceMSPC --model cycle_gan --no_dropout --batch_size 1