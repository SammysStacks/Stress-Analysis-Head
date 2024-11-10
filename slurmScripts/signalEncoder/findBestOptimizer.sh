#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')

numSignalEncoderLayers=8
waveletType='bior3.7'
encodedDimension=64
goldenRatio=4

for optimizer in "${optimizers[@]}"
do
    echo "Submitting job with $numSignalEncoderLayers numSignalEncoderLayers $goldenRatio goldenRatio $encodedDimension encodedDimension on $1 using $waveletType waveletType and $optimizer optimizer."

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
