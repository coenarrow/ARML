#!/bin/bash
#SBATCH --job-name=bcss3
#SBATCH --output=slurm_out/%A_bcss3.out
#SBATCH --error=slurm_out/%A_bcss3.err
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1

REPO_DIR="/group/pgh004/carrow/repo/ARML"
export PYTHONPATH=$PYTHONPATH:$REPO_DIR
cd $REPO_DIR
echo "Current Working Directory: $(pwd)"

module load cuda/12.4
module list

# Detect the operating system
OS=$(uname)

# Source conda for Linux/macOS or activate conda for Windows
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    # For Linux and macOS
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /group/pgh004/carrow/repo/ARML/env
else
    # For Windows
    eval "$(conda shell.bash hook)"
    conda activate ./env
fi

echo "Activated Conda Environment"
conda info --envs

echo "Using python: $(which python)"
python -m pip list | grep mxnet

echo "Start Time: $(date)"
python 3_train_stage2.py --dataset bcss --dataroot datasets/BCSS-WSSS --epochs 1
echo "End Time: $(date)"