#!/bin/bash

waveletTypes=('bior3.3')
signalEncoderLayers=(4 8 16 32)
encodedDimensions=(64 128 256)
goldenRatios=(1 2 4)
optimizers=('AdamW')

lr_general=0.1
lr_physio=0.01

for optimizer in "${optimizers[@]}"
do
  for waveletType in "${waveletTypes[@]}"
  do
    for encodedDimension in "${encodedDimensions[@]}"
    do
      for goldenRatio in "${goldenRatios[@]}"
      do
        for numSignalEncoderLayers in "${signalEncoderLayers[@]}"
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
    done
  done
done
