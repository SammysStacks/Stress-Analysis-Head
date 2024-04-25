#!/bin/bash

# Specify the module parameters
#SBATCH --mem-per-cpu=48G
#SBATCH --time=7-00:00:00   # Total runtime. Maximum 7 days = 128 hours
#SBATCH --gres=gpu:p100:1   # Number of GPUs on the node. Maximum of 4
#SBATCH --partition=gpu     # Specify the GPU partition (May change)
#SBATCH --ntasks=1    # number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Total number of GPU nodes

# Specify the job information
#SBATCH -J "AutoencoderModel compressionFactor $1 expansionFactor $2"      # The job name
#SBATCH --mail-user=ssolomon@caltech.edu   # Email any results or errors

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load in your modules
module load intel-oneapi-mkl/2024.0.0-oneapi-2023.2.1-4aoiyez
module load cuda/11.8.0-gcc-11.3.1-nlhqhb5
module load python/3.10.12-gcc-11.3.1-n4zmj3v   # Load in the latest python version
module load openmpi/5.0.1-gcc-11.3.1-j4o6ryt    # Load in openMPI for cross-node talk
module load cuda/12.2.1-gcc-11.3.1-yfdtcdo      # Load the CUDA module
module load nvhpc/23.7-gcc-11.3.1-gifa6ml
nvcc --version

# Set the PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN FILE
sh autoencoderTimeAnalysis.sh $1 $2 $3
