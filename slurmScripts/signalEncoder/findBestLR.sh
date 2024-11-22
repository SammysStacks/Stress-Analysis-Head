#!/bin/bash

lrs_general=(0.01)
lrs_physio=(0.1)

numSignalEncoderLayers=16
waveletType='bior3.3'
encodedDimension=256
optimizer='AdamW'
goldenRatio=2

for lr_physio in "${lrs_physio[@]}"
do
  for lr_general in "${lrs_general[@]}"
  do
    # Check if goldenRatio is greater than numSignalEncoderLayers
    if [ "$goldenRatio" -gt "$numSignalEncoderLayers" ]; then
      continue  # Skip this iteration if the condition is true
    fi

    echo "Submitting job with $numSignalEncoderLayers numSignalEncoderLayers, $goldenRatio goldenRatio, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer on $1"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
    else
        echo "No known device listed: $1"
    fi
    done
done
