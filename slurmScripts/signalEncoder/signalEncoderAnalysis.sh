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
    --profileDimension "${11}" \
    --beta1 "${12}" \
    --beta2 "${13}" \
    --momentum_decay "${14}" \
    --minAngularThreshold "${15}" \
    --maxAngularThreshold "${16}" \
    --profileWD "${19}" \

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
