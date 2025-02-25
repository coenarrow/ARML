#!/bin/bash
#SBATCH --job-name=t2f1
#SBATCH --output=slurm_out/%A_t2f1.out
#SBATCH --error=slurm_out/%A_t2f1.err
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
python 1_train_stage1.py --dataset t2f --trainroot datasets/t2f/train/ --testroot dataset/t2f/val/ --max_epoches 1 --batch_size 24
echo "End Time: $(date)"