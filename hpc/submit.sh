#!/usr/bin/env bash

#SBATCH --job-name=setup_pytorch

#SBATCH --mail-type=all
#SBATCH --mail-user=ehrletim@students.zhaw.ch

#SBATCH --time=00-02:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40s:1


# shellcheck disable=SC1091
source load_env.sh

hostname
# ## get GPU info
nvidia-smi

echo
echo "#########################################   PyTorch Info"
echo

python pytorch_info.py

echo
echo "#########################################   DL part"
echo