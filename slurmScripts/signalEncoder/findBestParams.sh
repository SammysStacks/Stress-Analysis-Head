#!/bin/bash

# General parameters: 320
allNumEncodedWeights=(4 8 16 32 64)  # 6
numSpecificEncoderLayers_arr=(1 2)  # 2
signalEncoderLayers_arr=(2 4 8 12 16)  # 5
encodedDimensions_arr=(64 128 256 512)  # 4

# General parameters: 18
allNumEncodedWeights=(32)  # 6
numSpecificEncoderLayers_arr=(1)  # 1
signalEncoderLayers_arr=(8)  # 3
encodedDimensions_arr=(256)  # 1
numProfileEpochs_arr=(15 10 5 20 25 30 35 40 45 50)  # 3

# Finalized parameters.
lrs_reversible=('1e-3')  # 5e-4 <= x <= 2e-3
lrs_profileGen=('1e-4') # '5e-5')  # 5e-5 <= x <= 1e-4

beta1s=('0.7' '0.8' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.9')  # 0.5 <= x <= 0.95
beta2s=('0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.99')  # 0.9 <= x <= 0.999
momentums=('1e-3' '1e-5' '1e-4' '1e-2' '1e-1' '0')  # 0.9 <= x <= 0.999

beta1s=('0.71')  # 0.5 <= x <= 0.95
beta2s=('0.91')  # 0.9 <= x <= 0.999
momentums=('0.0011')  # 0.9 <= x <= 0.999
numProfileEpochs_arr=(12)  # 3

lrs_profile=('0.2' '0.1' '0.15' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1')  # 0.05 <= x <= 2
wds_profileGen=('0' '1e-8' '1e-7' '1e-6' '1e-5' '1e-4' '1e-3' '1e-2' '1e-1' '1')  # 10
wds_reversible=('0' '1e-8' '1e-7' '1e-6' '1e-5' '1e-4' '1e-3' '1e-2' '1e-1' '1')  # 10
wds_profile=('0' '1e-8' '1e-7' '1e-6' '1e-5' '1e-4' '1e-3' '1e-2' '1e-1' '1')  # 10

# Weight decays: 27
wds_profileGen=('0')  #
wds_reversible=('0')  # 4
wds_profile=('0')  #
lrs_profile=('0.21')

# Finalized parameters.
waveletTypes_arr=('bior3.1')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'
optimizers_arr=('NAdam')  # 'AdamW'; RAdam was bad for retraining profile.

for beta1s in "${beta1s[@]}"
do
  for beta2s in "${beta2s[@]}"
  do
    for momentums in "${momentums[@]}"
    do
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
                                    sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums"
                                elif [ "$1" == "GPU" ]; then
                                    sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums"
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
    done
  done
done
