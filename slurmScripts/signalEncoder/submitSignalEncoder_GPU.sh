#!/bin/bash

# Specify the module parameters
#SBATCH --time=7-0:00:00   # Total runtime. Maximum 9-10 days, days-hours:minutes:seconds
#SBATCH --gres=gpu:p100:1   # Number of GPUs on the node. Maximum of 4
#SBATCH --mem-per-cpu=32G   # Memory per CPU core
#SBATCH --partition=gpu     # Specify the GPU partition (May change)
#SBATCH --ntasks=1    # number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Total number of nodes

# Specify the job information
#SBATCH --mail-user=ssolomon@caltech.edu    # Email any results or errors

# Specify dynamic paths for output and error files
#SBATCH --output="slurmOutputs/%x-%j.out"  # Standard output file

# Load in your modules
nvcc --version

# Set the PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RUN FILE
sh signalEncoderAnalysis.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}"