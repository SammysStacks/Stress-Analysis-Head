#!/bin/bash

# General parameters: 320
allNumEncodedWeights=(2 4 8 16 32)  # 8
numSpecificEncoderLayers_arr=(1 2)  # 2
signalEncoderLayers_arr=(2 4 8 12 16)  # 5
encodedDimensions_arr=(64 128 256 512)  # 4

# General parameters: 18
allNumEncodedWeights=(32)  # 6
numSpecificEncoderLayers_arr=(1)  # 1
signalEncoderLayers_arr=(8)  # 3
encodedDimensions_arr=(256)  # 1
numProfileEpochs_arr=(20 10)  # 1

# Learning rates: 27
lrs_reversible=('1e-4' '1e-3' '1e-2')  # 3
lrs_profile=('10' '100' '1' '0.1')  # 3
lrs_profileGen=('1e-4' '1e-5' '1e-3' '1e-2')  # 3

# Weight decays: 27
wds_reversible=('0' '1e-4' '1e-2')  # 4
wds_profile=('0' '0.1' '1e-3')  # 4
wds_profileGen=('0' '1e-4' '1e-2')  # 4

wds_reversible=('0')  # 4
wds_profile=('0')  # 4
wds_profileGen=('0')  # 4

# Finalized parameters.
waveletTypes_arr=('bior3.1')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'
optimizers_arr=('RAdam')

for numEncodedWeights in "${allNumEncodedWeights[@]}"
do
  for numProfileEpochs in "${numProfileEpochs_arr[@]}"
  do
    for lr_profile in "${lrs_profile[@]}"
    do
      for wd_reversible in "${wds_reversible[@]}"
      do
        for wd_profile in "${wds_profile[@]}"
        do
          for lr_reversible in "${lrs_reversible[@]}"
          do
            for wd_profileGen in "${wds_profileGen[@]}"
            do
              for lr_profileGen in "${lrs_profileGen[@]}"
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

                          echo "Submitting job with $numSharedEncoderLayers numSharedEncoderLayers, $numSpecificEncoderLayers numSpecificEncoderLayers, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer, $lr_profile lr_profile, $lr_reversible lr_reversible"

                          if [ "$1" == "CPU" ]; then
                              sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen"
                          elif [ "$1" == "GPU" ]; then
                              sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen"
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
    done
  done
done