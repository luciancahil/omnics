#!/bin/sh
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --job-name=CGCNN
#SBATCH --account=st-singha53-1-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=royhe@student.ubc.ca
#SBATCH --output=CGCNN.txt
#SBATCH --error=CGCNN_error.txt

bash run_bo.sh train
