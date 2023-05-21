#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 CUT/train.py --dataroot Dataset/Combined --name real2viceCUT_V1 --CUT_mode CUT --direction BtoA --continue_train --epoch_count 101