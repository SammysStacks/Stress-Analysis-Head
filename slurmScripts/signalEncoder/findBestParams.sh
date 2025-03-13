#!/bin/bash

# Optimizer parameters.
optimizers_arr=('NAdam')  # NAdam > AdamW > RAdam >= Adam > Adamax
momentums_arr=('0.001')
beta1s_arr=('0.7')
beta2s_arr=('0.8')

# Learning parameters.
lrs_profile=(0.02)  # 0.005 <= x <= 0.075
lrs_profileGen=('1e-4') # x <= 1e-4;
lrs_reversible=('1e-3')  # 1e-4 <= x == 1e-3 -> [2.5e-4, 5e-4]

# Known interesting parameters: 63
numSharedEncoderLayers_arr=(0 1 2 3 4 5 6 7 8 9)  # 9
numSpecificEncoderLayers_arr=(0 1 2)  # 7

# Known interesting parameters: 7*7 = 49
encodedDimensions_arr=(128 256 512 1024)  # 7
profileParams=(64 128 256 512 1024)  # 7

# Single switchable: 7
numProfileShots_arr=(24)  # 6; 12 <= x <= 24

# Neural operator parameters.
waveletTypes_arr=(
    # 15 rbio wavelets
    'rbio1.1' 'rbio1.3' 'rbio1.5' 'rbio2.2' 'rbio2.4' 'rbio2.6' 'rbio2.8' \
    'rbio3.1' 'rbio3.3' 'rbio3.5' 'rbio3.7' 'rbio3.9' 'rbio4.4' 'rbio5.5' 'rbio6.8' \

    # 20 sym wavelets
    'sym2' 'sym3' 'sym4' 'sym5' 'sym6' 'sym7' 'sym8' 'sym9' 'sym10' \
    'sym11' 'sym12' 'sym13' 'sym14' 'sym15' 'sym16' 'sym17' 'sym18' 'sym19' 'sym20' \

    # Miscellaneous wavelets
    'haar'

    # 15 bior wavelets
    'bior1.1' 'bior1.3' 'bior1.5' 'bior2.2' 'bior2.4' 'bior2.6' 'bior2.8' \
    'bior3.1' 'bior3.3' 'bior3.5' 'bior3.7' 'bior3.9' 'bior4.4' 'bior5.5' 'bior6.8' \
    # 'bior3.1' > 'bior3.3' > 'bior2.2' > 'bior3.5'

    # 17 coif wavelets
    'coif1' 'coif2' 'coif3' 'coif4' 'coif5' 'coif6' 'coif7' 'coif8' 'coif9' 'coif10' \
    'coif11' 'coif12' 'coif13' 'coif14' 'coif15' 'coif16' 'coif17' \
)

# Angular reference states.
minAngularThresholds=(0.01 0.05 0.075 0.1)  # [0.01, 0.25]
maxAngularThresholds=(45)

# Binary reference states.
numSpecificEncoderLayers_arr=(1)
numSharedEncoderLayers_arr=(15 13 11 9 7)  # [4, 10]; Best: 5 and 7

# Binary reference states.
encodedDimensions_arr=(512)
profileParams=(128)

# Reference states.
waveletTypes_arr=('bior3.1')
numProfileShots_arr=(24)  # (8, [16, 24], 32)

for minAngularThreshold in "${minAngularThresholds[@]}"
do
    for maxAngularThreshold in "${maxAngularThresholds[@]}"
    do
        for beta1s in "${beta1s_arr[@]}"
        do
            for beta2s in "${beta2s_arr[@]}"
            do
                for momentums in "${momentums_arr[@]}"
                do
                    for profileDimension in "${profileParams[@]}"
                    do
                        for numProfileShots in "${numProfileShots_arr[@]}"
                        do
                            for lr_profile in "${lrs_profile[@]}"
                            do
                                for lr_reversible in "${lrs_reversible[@]}"
                                do
                                    for lr_profileGen in "${lrs_profileGen[@]}"
                                    do
                                        for optimizer in "${optimizers_arr[@]}"
                                        do
                                            for waveletType in "${waveletTypes_arr[@]}"
                                            do
                                                for encodedDimension in "${encodedDimensions_arr[@]}"
                                                do
                                                    for numSharedEncoderLayers in "${numSharedEncoderLayers_arr[@]}"
                                                    do
                                                        for numSpecificEncoderLayers in "${numSpecificEncoderLayers_arr[@]}"
                                                        do
                                                            if (( encodedDimension < profileDimension )); then
                                                            echo "Encoded dimension is less than profile dimension."
                                                                continue
                                                            fi

                                                            if (( $(echo "$maxAngularThreshold <= $minAngularThreshold" | bc -l) )); then
                                                            echo "Angular threshold max is less than or equal to angular threshold min."
                                                                continue
                                                            fi

                                                            filename="signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_numProfileShots_${numProfileShots}_optimizer_${optimizer}_profileDim_${profileDimension}_minAngleThresh_${minAngularThreshold}_maxAngleThresh_${maxAngularThreshold}"

                                                            if [ "$1" == "CPU" ]; then
                                                                sbatch -J "$filename" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$profileDimension" "$beta1s" "$beta2s" "$momentums" "$minAngularThreshold" "$maxAngularThreshold"
                                                            elif [ "$1" == "GPU" ]; then
                                                                sbatch -J "$filename" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$profileDimension" "$beta1s" "$beta2s" "$momentums" "$minAngularThreshold" "$maxAngularThreshold"
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
