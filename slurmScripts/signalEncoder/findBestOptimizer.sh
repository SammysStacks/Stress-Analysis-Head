#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')
optimizers=('Adam' 'AdamW' 'NAdam' 'RMSprop' 'Adadelta' 'Adadelta')  # 7 optimizers.  'RAdam'

numSpecificEncoderLayers=2
numSharedEncoderLayers=16
waveletType='bior3.3'
encodedDimension=256

lr_general=0.001
lr_physio=0.01

for optimizer in "${optimizers[@]}"
do
    echo "Submitting job with $numSharedEncoderLayers numSharedEncoderLayers, $numSpecificEncoderLayers numSpecificEncoderLayers, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer on $1"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSharedEncoderLayers_${numSharedEncoderLayers}_numSpecificEncoderLayers_${numSpecificEncoderLayers}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSharedEncoderLayers" "$numSpecificEncoderLayers" "$encodedDimension" "$1" "$waveletType" "$optimizer" "$lr_physio" "$lr_general"
    else
        echo "No known device listed: $1"
    fi
done
