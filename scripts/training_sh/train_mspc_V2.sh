#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 MSPC/train.py --dataroot Dataset/Combined_V2 --name real2viceMSPC_V2 --model cycle_gan --direction AtoB --dataset_mode unaligned --continue_train --epoch_count 126