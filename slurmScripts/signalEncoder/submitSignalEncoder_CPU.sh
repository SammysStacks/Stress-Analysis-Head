#!/bin/bash

# Specify the module parameters
#SBATCH --time=7-00:00:00   # Total runtime. Maximum 9-10 days
#SBATCH --mem=24G    # Total memory
#SBATCH --ntasks=1    # number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Total number of nodes

# Specify the job information
#SBATCH --mail-user=ssolomon@caltech.edu    # Email any results or errors

# Specify dynamic paths for output and error files
#SBATCH --output="slurmOutputs/%x-%j.out"  # Standard output file

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN,END,FAIL

# Load in your modules
module load intel-oneapi-mkl/2023.2.0-gcc-11.3.1-6dhawvh
module load intel-oneapi-mkl/2023.2.0-gcc-13.2.0-ohvyk7g
module load openmpi/4.1.5-gcc-13.2.0-24q3ap2    # Load in openMPI for cross-node talk

# RUN FILE
sh signalencoderGroupAnalysis.sh "$1" "$2" "$3" "$4" "$5" "$6"