#!/bin/bash

# General parameters: 18
allNumEncodedWeights=(8 16 32 64 128 256)  # 6
numSpecificEncoderLayers_arr=(1)  # 1
signalEncoderLayers_arr=(4 8 12)  # 3
encodedDimensions_arr=(256)  # 1

# Learning rates: 6
lrs_general=('1e-3' '1e-4')  # 2
lrs_physio=('1e-1' '1e-2' '1e-3')  # 3

# Finalized parameters.
waveletTypes_arr=('bior3.1')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'
optimizers_arr=('RAdam')

for numEncodedWeights in "${allNumEncodedWeights[@]}"
do
  for lr_physio in "${lrs_physio[@]}"
  do
    for lr_general in "${lrs_general[@]}"
    do
      for optimizer in "${optimizers_arr[@]}"
      do
        for waveletType in "${waveletTypes_arr[@]}"
        do
          for encodedDimension in "${encodedDimensions_arr[@]}"
          do
            for numSpecificEncoderLayers in "${numSpecificEncoderLayers_arr[@]}"
            do
              for numSharedEncoderLayers in "${signalEncoderLayers_arr[@]}"
              do
                # Check if numSpecificEncoderLayers is greater than half the numSharedEncoderLayers
                if [ $((2 * numSpecificEncoderLayers)) -gt "$numSharedEncoderLayers" ]; then
                  continue  # Skip this iteration if the condition is true
                fi

                echo "Submitting job with $numSharedEncoderLayers numSharedEncoderLayers, $numSpecificEncoderLayers numSpecificEncoderLayers, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer, $lr_physio lr_physio, $lr_general lr_general"

                if [ "$1" == "CPU" ]; then
                    sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$numEncodedWeights"
                elif [ "$1" == "GPU" ]; then
                    sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$numEncodedWeights"
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
done
