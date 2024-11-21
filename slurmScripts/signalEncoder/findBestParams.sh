#!/bin/bash

waveletTypes=('bior3.7' 'bior6.8')
signalEncoderLayers=(16 32 48 64)
encodedDimensions=(128 256)
optimizers=('AdamW')
goldenRatios=(1 2 4)

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
              sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
          elif [ "$1" == "GPU" ]; then
              sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
          else
              echo "No known device listed: $1"
          fi
        done
      done
    done
  done
done
