#!/bin/bash

lrs_general=(0.001 0.0001)
lrs_physio=(0.1 0.01)

numSpecificEncoderLayers=2
numSharedEncoderLayers=9
waveletType='bior3.3'
encodedDimension=256
optimizer='RAdam'

for lr_physio in "${lrs_physio[@]}"
do
  for lr_general in "${lrs_general[@]}"
  do
    # Check if numSpecificEncoderLayers is greater than numSharedEncoderLayers
    if (( $(echo "$lr_physio < $lr_general" | bc -l) )); then
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
