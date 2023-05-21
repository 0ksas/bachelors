#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 CUT/train.py --dataroot Dataset/Combined_V2 --name real2viceCUT_V2 --CUT_mode CUT