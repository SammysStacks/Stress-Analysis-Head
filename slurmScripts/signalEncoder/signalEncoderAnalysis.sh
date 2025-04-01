#!/bin/sh
start=$(date +%s)

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numSharedEncoderLayers "$1" \
    --numSpecificEncoderLayers "$2" \
    --encodedDimension "$3" \
    --numProfileShots "$4" \
    --deviceListed "HPC-$5" \
    --submodel "signalEncoderModel" \
    --waveletType "$6" \
    --optimizerType "$7" \
    --profileLR "$8" \
    --reversibleLR "$9" \
    --physGenLR "${10}" \
    --beta1 "${11}" \
    --beta2 "${12}" \
    --momentum_decay "${13}" \
    --minAngularThreshold "${14}" \
    --maxAngularThreshold "${15}" \
    --numIgnoredSharedHF "${16}" \

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
