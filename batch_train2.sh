#!/bin/bash
#SBATCH --job-name=voxel_AE_train_step1      # Job name
#SBATCH --output=/home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/log/slurm_%j.out
#SBATCH --error=/home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/log/slurm_%j.err
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=8                   # CPU cores per task
#SBATCH --mem=32G                           # Memory
#SBATCH --time=60:00:00                     # Max runtime hh:mm:ss
#SBATCH --mail-type=END,FAIL                # Email notification on end/fail
#SBATCH --mail-user=leonardo.monrroy@kemi.uu.se.com  # Your email
#SBATCH --account=naiss2025-22-1083


source ~/.bashrc
conda activate diffsaxs

# Navigate to working directory
cd /home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/

# Run training
python3 train_network_2.py


