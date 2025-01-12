#!/bin/bash

# Optimizer parameters.
optimizers_arr=('NAdam' 'AdamW')  # AdamW == NAdam > RAdam >= Adam > Adamax
momentums_arr=('0.01')  # Removed from filename
beta1s_arr=('0.7')  # Removed from filename
beta2s_arr=('0.8')  # Removed from filename

# Weight decay parameters.
wds_profile=('1e-6')  # 1e-6 ==> x <== 1e-3; Removed from filename
wds_profileGen=('1e-5')  # 1e-5 == x <= 1e-4; Removed from filename
wds_reversible=('1e-4')  # 1e-4 == x <= 1e-3; Removed from filename

# Known interesting parameters: 96
numSharedEncoderLayers_arr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)  # 16
numSpecificEncoderLayers_arr=(1 2 3 4 5 6)  # 6

# Known interesting parameters: 144
profileParams=(16 32 64 128 256 512)  # 6
numProfileShots_arr=(4 8 12 16 24 32)  # 6
encodedDimensions_arr=(32 64 128 256 512)  # 4
initialProfileAmp_arr=('0.01')  # 0.005 <= x <= 0.05

# Neural operator parameters.
waveletTypes_arr=(
    # 15 bior wavelets
    'bior1.1' 'bior1.3' 'bior1.5' 'bior2.2' 'bior2.4' 'bior2.6' 'bior2.8' \
    'bior3.1' 'bior3.3' 'bior3.5' 'bior3.7' 'bior3.9' 'bior4.4' 'bior5.5' 'bior6.8' \
)  # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'

# Learning parameters.
lrs_profile=('0.001' '0.005' '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1')  # 0.005 <= x <= 0.05
lrs_profileGen=('1e-4') # # 5e-5 <= x == 1e-4; Removed from filename
lrs_reversible=('1e-3')  # 1e-4 <= x == 1e-3; Removed from filename

# Finished
encodedDimensions_arr=(128)
waveletTypes_arr=('bior3.1')

# Collective Switchables: 20
#numSpecificEncoderLayers_arr=(1 2 3 4)
#numSharedEncoderLayers_arr=(2 4 5 8 12 16)
numSpecificEncoderLayers_arr=(2)
numSharedEncoderLayers_arr=(6)

# Collective Switchables: 5
#numProfileShots_arr=(32 20 16 12 8)
numProfileShots_arr=(16)

# Collective Switchables: 4
#profileParams=(16 64 128 256)
profileParams=(128)

# Collective Switchables: 90
lrs_profile=('0.04' '0.051' '0.06' '0.07' '0.08' '0.09' '0.1')  # 0.005 <= x <= 0.05
lrs_reversible=('3e-4' '1e-4' '5e-4' '1e-3' '1e-2')  # 1e-4 <= x == 1e-3;
lrs_profileGen=('1e-4' '2e-4' '3e-4') # # 1e-4 <= x == 1e-3; lrs_profileGen <= lrs_reversible
#lrs_profile=('0.05')
#lrs_reversible=('3e-4')
#lrs_profileGen=('1e-4')

# Single Switchables: 2
#optimizers_arr=('Adam' 'RAdam')  # AdamW == NAdam > RAdam > Adam > Adamax
optimizers_arr=('AdamW' 'NAdam')  # AdamW == NAdam > RAdam > Adam > Adamax

# Weight decay parameters.
wds_profile=('1e-2' '1e-4' '1e-6' '1e-8')  # 1e-6 ==> x <== 1e-3; Removed from filename
wds_reversible=('1e-6' '1e-3' '1e-2' '0')  # 1e-4 == x <= 1e-3; Removed from filename
wds_profileGen=('1e-6' '1e-4' '1e-3' '0')  # 1e-5 == x <= 1e-4; Removed from filename
wds_profile=('0')  
wds_reversible=('1e-4') 
wds_profileGen=('1e-2') 

#momentums_arr=('0.004' '0.01' '0.025' '0.001' '0.0025' '0.0075')  # Removed from filename
#beta1s_arr=('0.7' '0.8' '0.6' '0.5')  # Removed from filename
#beta2s_arr=('0.8' '0.9' '0.99' '0.95' '0.999')  # Removed from filename

for beta1s in "${beta1s_arr[@]}"
do
  for beta2s in "${beta2s_arr[@]}"
  do
    for momentums in "${momentums_arr[@]}"
    do
      for profileDimension in "${profileParams[@]}"
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
                                for numSharedEncoderLayers in "${numSharedEncoderLayers_arr[@]}"
                                do
                                  if (( encodedDimension < profileDimension )); then
                                      continue
                                  fi
                                  
                                  if [ "$1" == "CPU" ]; then
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$profileDimension" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$initialProfileAmp"
                                  elif [ "$1" == "GPU" ]; then
                                      sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$profileDimension" "$wd_profile" "$wd_reversible" "$wd_profileGen" "$beta1s" "$beta2s" "$momentums" "$initialProfileAmp"
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
