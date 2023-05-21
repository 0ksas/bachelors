#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1

source cut_gpu_env/bin/activate
python3.8 CUT/test.py --dataroot Dataset/Combined --name real2viceCUT_V1 --CUT_mode CUT --no_dropout --direction BtoA 