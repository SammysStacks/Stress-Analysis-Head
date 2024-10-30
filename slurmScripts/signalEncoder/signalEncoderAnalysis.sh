#!/bin/sh
start=$(date +%s)

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numSignalEncoderLayers "$1" \
    --goldenRatio "$2" \
    --encodedDimension "$3" \
    --deviceListed "HPC-$4" \
#    --submodel modelConstants.signalEncoderModel \
#    --waveletType "$5" \
#    --optimizerType "$6" \

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
