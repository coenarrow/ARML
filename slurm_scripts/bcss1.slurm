#!/bin/bash
#SBATCH --job-name=bcss1
#SBATCH --output=slurm_out/%A_bcss1.out
#SBATCH --error=slurm_out/%A_bcss1.err
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pgh004/carrow/repo/ARML/env

echo "Activated Conda Environment"
conda info --envs

echo "Using python: $(which python)"
python -c "import mxnet; print('mxnet imported successfully')"

echo "Start Time: $(date)"
python 1_train_stage1.py --dataset bcss --trainroot datasets/BCSS-WSSS/train/ --testroot dataset/BCSS-WSSS/test/ --max_epoches 1
echo "End Time: $(date)"