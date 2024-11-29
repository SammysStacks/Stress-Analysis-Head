#!/bin/bash

waveletTypes=('bior3.1' 'bior3.3' 'bior3.5')
numSpecificEncoderLayers=(1 2 4)
signalEncoderLayers=(4 8 12 16)
lrs_general=(0.001 0.0001)
encodedDimensions=(256)
lrs_physio=(0.1 0.01)
optimizers=('RAdam')

for lr_physio in "${lrs_physio[@]}"
do
  for lr_general in "${lrs_general[@]}"
  do
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
  done
done
