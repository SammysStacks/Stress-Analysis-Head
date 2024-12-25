#!/bin/bash

# Optimizer parameters.
optimizers_arr=('NAdam' 'AdamW' 'RAdam' 'Adam')  # AdamW == NAdam > RAdam > Adam > Adamax
momentums_arr=('0.004')
beta1s_arr=('0.7')
beta2s_arr=('0.9')

# Weight decay parameters.
wds_profile=('1e-6')  # 1e-6 ==> x <== 1e-3
wds_profileGen=('1e-5')  # 1e-5 == x <= 1e-4
wds_reversible=('1e-4')  # 1e-4 == x <= 1e-3

# Known interesting parameters: 320
signalEncoderLayers_arr=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)  # 16
allNumEncodedWeights=(4 8 16 32 64 128 256 512)  # 5
numSpecificEncoderLayers_arr=(1 2 3 4)  # 4
encodedDimensions_arr=(64 128 256 512)  # 4
numProfileShots_arr=(1 2 3 4 8 16 24 32)  # 7
initialProfileAmp_arr=('0.01')  # 0.005 <= x <= 0.05

# Neural operator parameters.
waveletTypes_arr=('bior3.1' 'bior3.3' 'bior3.5' 'bior2.2')  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'

# Learning parameters: 3
lrs_profile=('0.001' '0.005' '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1')  # 0.01 <= x <= 0.05
lrs_profileGen=('1e-4') # # 5e-5 <= x == 1e-4
lrs_reversible=('1e-3')  # 1e-4 <= x == 1e-3

# Single Switchable.
numSpecificEncoderLayers_arr=(1)
waveletTypes_arr=('bior3.1')
signalEncoderLayers_arr=(8)
encodedDimensions_arr=(256)
allNumEncodedWeights=(128)
numProfileShots_arr=(16)
optimizers_arr=('NAdam')
lrs_profile=('0.01')

for beta1s in "${beta1s_arr[@]}"
do
  for beta2s in "${beta2s_arr[@]}"
  do
    for momentums in "${momentums_arr[@]}"
    do
      for numEncodedWeights in "${allNumEncodedWeights[@]}"
      do
        for initialProfileAmp in "${initialProfileAmp_arr[@]}"
        do
          for numProfileShots in "${numProfileShots_arr[@]}"
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
                                  if (( encodedDimension < numEncodedWeights )); then
                                      continue
                                  fi

                                  if (( numSharedEncoderLayers < 2*numSpecificEncoderLayers )); then
                                      continue
                                  fi

                                  if [ "$1" == "CPU" ]; then
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$initialProfileAmp"
                                  elif [ "$1" == "GPU" ]; then
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$numEncodedWeights" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$initialProfileAmp"
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
