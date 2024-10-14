#!/bin/sh
start=$(date +%s)

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numSigLiftedChannels "$1" \
    --numSigEncodingLayers "$2" \
    --encodedSamplingFreq "$3" \
    --deviceListed "HPC-$4" \
    --submodel modelConstants.signalEncoderModel \
    --signalEncoderWaveletType "$5" \
    --optimizerType "$6" \

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
