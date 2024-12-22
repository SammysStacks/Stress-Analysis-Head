#!/bin/bash

# Optimizer parameters.
optimizers_arr=('NAdam' 'RAdam' 'Adam' 'AdamW' 'Adamax')  # 'AdamW'; RAdam was bad for retraining profile.
momentums_arr=('0.004')
beta1s_arr=('0.7')
beta2s_arr=('0.9')

# Weight decay parameters.
wds_profile=('1e-6' '1e-3')  # 0 <= x <= 1e-2
wds_profileGen=('1e-5')  # 1e-5 <= x <= 1e-4
wds_reversible=('1e-4')  # 1e-4 <= x <= 1e-3

# General parameters: 320
uniformWeightLimits_arr=('0.01' '0.2' '0.1' '0.005' '0.001')  #
signalEncoderLayers_arr=(6)  # 1 2 3 4 5 6 7 8
encodedDimensions_arr=(32 64 128 256)  # 64 128 256 512
allNumEncodedWeights=(64 128 256)  # 4 8 16 32 64
numSpecificEncoderLayers_arr=(1)  # 1 2
numProfileEpochs_arr=(16 36)  # 3

# Neural operator parameters.
waveletTypes_arr=('bior3.1')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'

# Learning parameters.
lrs_profile=('0.01' '0.05' '0.1')  # 0.05 <= x <= 0.4
lrs_profileGen=('1e-4') # '5e-5')  # 5e-5 <= x <= 1e-4
lrs_reversible=('1e-3')  # 5e-4 <= x <= 2e-3

optimizers_arr=('NAdam' 'RAdam' 'Adamax')  # 'AdamW'; RAdam was bad for retraining profile.
encodedDimensions_arr=(256)  # 64 128 256 512

for beta1s in "${beta1s_arr[@]}"
do
  for beta2s in "${beta2s_arr[@]}"
  do
    for momentums in "${momentums_arr[@]}"
    do
      for numEncodedWeights in "${allNumEncodedWeights[@]}"
      do
        for uniformWeightLimits in "${uniformWeightLimits_arr[@]}"
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
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$uniformWeightLimits"
                                  elif [ "$1" == "GPU" ]; then
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileEpochs" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$uniformWeightLimits"
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
done
