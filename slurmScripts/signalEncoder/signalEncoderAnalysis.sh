#!/bin/sh
start=$(date +%s)

# Get the value of signalEncoderModel from modelConstants.py
signalEncoderModel=$(python -c "from modelConstants import signalEncoderModel; print(signalEncoderModel)")

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numSignalEncoderLayers "$1" \
    --goldenRatio "$2" \
    --encodedDimension "$3" \
    --deviceListed "HPC-$4" \
    --submodel "$signalEncoderModel" \
    --waveletType "$5" \
#    --optimizerType "$6"

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
