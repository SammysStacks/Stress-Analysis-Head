#!/bin/bash

encodedDimensions=(64 128 256 512)
# Total: 4

goldenRatios=(8 16)
# Total: 2

signalEncoderLayers=(16 32)
# Total: 2

waveletType="bior3.7"
optimizer="AdamW"     # Replace with actual value

for encodedDimension in "${encodedDimensions[@]}"
do
  for goldenRatio in "${goldenRatios[@]}"
  do
    for numSignalEncoderLayers in "${signalEncoderLayers[@]}"
    do
      echo "Submitting job with $numSignalEncoderLayers numSignalEncoderLayers, $goldenRatio goldenRatio, $encodedDimension encodedDimension on $1"

      if [ "$1" == "CPU" ]; then
          sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType"
      elif [ "$1" == "GPU" ]; then
          sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType"
      else
          echo "No known device listed: $1"
      fi
    done
  done
done
