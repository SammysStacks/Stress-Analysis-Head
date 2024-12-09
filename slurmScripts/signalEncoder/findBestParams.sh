#!/bin/bash

# General parameters: 320
allNumEncodedWeights=(2 4 8 16 32)  # 8
numSpecificEncoderLayers_arr=(1 2)  # 2
signalEncoderLayers_arr=(2 4 8 12 16)  # 5
encodedDimensions_arr=(64 128 256 512)  # 4

# General parameters: 18
allNumEncodedWeights=(64)  # 6
numSpecificEncoderLayers_arr=(1)  # 1
signalEncoderLayers_arr=(8)  # 3
encodedDimensions_arr=(256)  # 1

# Learning rates: 9
lrs_general=('1e-2' '1e-3' '1e-4')  # 3
lrs_physio=('0.1' '0.33' '0.01')  # 3

# Weight decays: 25
wds_general=('0' '1e-4' '1e-3' '1e-2' '1e-1')  # 4
wds_physio=('0' '1e-4' '1e-3' '1e-2' '1e-1')  # 4

# Finalized parameters.
waveletTypes_arr=('bior3.1')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'
optimizers_arr=('RAdam')


for numEncodedWeights in "${allNumEncodedWeights[@]}"
do
  for lr_physio in "${lrs_physio[@]}"
  do
    for wd_general in "${wds_general[@]}"
    do
      for wd_physio in "${wds_physio[@]}"
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
                        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$numEncodedWeights" "$wd_general" "$wd_physio"
                    elif [ "$1" == "GPU" ]; then
                        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general" "$numEncodedWeights" "$wd_general" "$wd_physio"
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
  done
done
