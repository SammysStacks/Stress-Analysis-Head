#!/bin/bash

# Optimizer parameters.
optimizers_arr=('NAdam')  # NAdam > AdamW > RAdam >= Adam > Adamax
momentums_arr=('0.001')
beta1s_arr=('0.7')
beta2s_arr=('0.8')

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

# Learning parameters.
lrs_profile=(0.01)  # 0.005 <= x <= 0.05
lrs_profileGen=('4e-4') # x <= 1e-4;
lrs_reversible=(0.05 0.025 0.075)  # [0.025, 0.075]

# Angular reference states.
minAngularThresholds=(1 0.25)  # 0.2 - 0.5 - 1
maxAngularThresholds=(90 20)  # [10, 45, 90]; Absolute min [6, 10]

# Binary reference states.
numSpecificEncoderLayers_arr=(1)
numSharedEncoderLayers_arr=(7)  # [4, 10]; Best: 5 and 7

# Profile parameters.
numProfileShots_arr=(32)  # (8, [16, 24], 32), (3)
encodedDimensions_arr=(256)  # [128, 256, 512]

# Wavelet states.
waveletTypes_arr=('bior3.1')
numIgnoredSharedHFs=(0)

for numIgnoredSharedHF in "${numIgnoredSharedHFs[@]}"
do
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
                                                            filename="signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_numProfileShots_${numProfileShots}_optimizer_${optimizer}_minAngleThresh_${minAngularThreshold}_maxAngleThresh_${maxAngularThreshold}_waveletType_${waveletType}_lr_reversible_${lr_reversible}_numIgnoredSharedHF_${numIgnoredSharedHF}_lr_profileGen_${lr_profileGen}"

                                                            if [ "$1" == "CPU" ]; then
                                                                sbatch -J "$filename" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$beta1s" "$beta2s" "$momentums" "$minAngularThreshold" "$maxAngularThreshold" "$numIgnoredSharedHF"
                                                            elif [ "$1" == "GPU" ]; then
                                                                sbatch -J "$filename" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$numProfileShots" "$1" "$waveletType" "$optimizer" "$lr_profile" "$lr_reversible" "$lr_profileGen" "$beta1s" "$beta2s" "$momentums" "$minAngularThreshold" "$maxAngularThreshold" "$numIgnoredSharedHF"
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
