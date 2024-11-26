#!/bin/bash

lrs_general=(0.001 0.0001 0.00001)
lrs_physio=(0.01 0.001)

numSpecificEncoderLayers=2
numSharedEncoderLayers=16
waveletType='bior2.2'
encodedDimension=256
maxNumDecompLevel=0
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
        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$maxNumDecompLevel"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$maxNumDecompLevel"
    else
        echo "No known device listed: $1"
    fi
    done
done
