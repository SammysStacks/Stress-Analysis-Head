#!/bin/bash

#SBATCH --mail-user=ssolomon@caltech.edu   # Email any results or errors
#SBATCH --partition=expansion  # Specify the expansion partition (May change)
#SBATCH --time=99:00:00        # Total runtime. Maximum 7 days = 128 hours
#SBATCH --ntasks=1  # The number of tasks per CPU
#SBATCH --nodes=1   # Total number of CPU nodes
#SBATCH --mem=386G  # Total CPU memory
#SBATCH -J "QM9"    # The job name

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# RUN FILE
sh autoencoderTimeAnalysis.sh $1 $2 $3
