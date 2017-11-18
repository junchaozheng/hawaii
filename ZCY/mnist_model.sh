#!/bin/sh
#

#SBATCH --job-name=0304_model
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4
 
cd /scratch/sy1743/A1/code/ 
module load python/intel/2.7.12
module load torch/intel/20170104
module load pytorch/intel/20170125
pip install --user torchvision
python mnist_model.py
