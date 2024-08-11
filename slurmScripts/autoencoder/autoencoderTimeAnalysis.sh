#!/bin/sh
start=$(date +%s)

# Pass the compressionFactor to the Python script
srun accelerate launch ./../../metaTrainingControl.py --compressionFactor "$1" --expansionFactor "$2" --deviceListed "HPC-$3" --submodel modelConstants.autoencoderModel

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"

