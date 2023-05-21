#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 MSPC/train.py --dataroot Dataset/CarDataset_0.5 --name real2viceCar --model cycle_gan --direction AtoB --dataset_mode unaligned --continue_train --epoch_count 50