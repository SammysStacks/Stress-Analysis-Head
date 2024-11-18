#!/bin/bash

# Specify the module parameters
#SBATCH --time=1-00:00:00   # Total runtime. Maximum 9-10 days
#SBATCH --gres=gpu:p100:1   # Number of GPUs on the node. Maximum of 4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu     # Specify the GPU partition (May change)
#SBATCH --ntasks=1    # number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Total number of nodes

# Specify the job information
#SBATCH --mail-user=ssolomon@caltech.edu    # Email any results or errors

# Specify dynamic paths for output and error files
#SBATCH --output="slurmOutputs/%x-%j.out"  # Standard output file

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN,END,FAIL

# Load in your modules
nvcc --version

# Set the PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN FILE
sh signalEncoderAnalysis.sh "$1" "$2" "$3" "$4" "$5" "$6"
