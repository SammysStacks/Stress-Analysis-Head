#!/bin/bash

numSpecificEncoderLayers=(1 2)
signalEncoderLayers=(4 8 12 16)
encodedDimensions=(256)
waveletTypes=('bior2.2' 'bior3.1' 'bior3.3' 'bior3.5' 'bior3.7')
optimizers=('RAdam')

lr_general=0.001
lr_physio=0.01

for optimizer in "${optimizers[@]}"
do
  for waveletType in "${waveletTypes[@]}"
  do
    for encodedDimension in "${encodedDimensions[@]}"
    do
      for numSpecificEncoderLayers in "${numSpecificEncoderLayers[@]}"
      do
        for numSharedEncoderLayers in "${signalEncoderLayers[@]}"
        do
          # Check if numSpecificEncoderLayers is greater than half the numSharedEncoderLayers
          if [ $((2 * numSpecificEncoderLayers)) -gt "$numSharedEncoderLayers" ]; then
            continue  # Skip this iteration if the condition is true
          fi

          echo "Submitting job with $numSharedEncoderLayers numSharedEncoderLayers, $numSpecificEncoderLayers numSpecificEncoderLayers, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer on $1"

          if [ "$1" == "CPU" ]; then
              sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
          elif [ "$1" == "GPU" ]; then
              sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
          else
              echo "No known device listed: $1"
          fi
        done
      done
    done
  done
done
