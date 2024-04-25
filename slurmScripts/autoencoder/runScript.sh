#!/bin/bash

expansionFactorStart=1.25
expansionFactorStep=0.25
expansionFactorEnd=2

compressionFactorStart=1.25
compressionFactorStep=0.25
compressionFactorEnd=2

for expansionFactor in $(seq $expansionFactorStart $expansionFactorStep $expansionFactorEnd)
do
    for compressionFactor in $(seq $compressionFactorStart $compressionFactorStep $compressionFactorEnd)
    do
        echo "Submitting job with $compressionFactor compressionFactor and $expansionFactor expansionFactor"
        
        if [ "$1" == "CPU" ]; then
            sbatch submitAutoencoder_CPU.sh $compressionFactor $expansionFactor $1
        elif [ "$1" == "GPU" ]; then
            sbatch submitAutoencoder_GPU.sh $compressionFactor $expansionFactor $1
        else
            echo "No known device listed: $1"
        fi
    done
done
