#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')

numSignalEncoderLayers=16
waveletType='bior3.3'
encodedDimension=256
goldenRatio=2

for optimizer in "${optimizers[@]}"
do
    # Check if goldenRatio is greater than numSignalEncoderLayers
    if [ "$goldenRatio" -gt "$numSignalEncoderLayers" ]; then
      continue  # Skip this iteration if the condition is true
    fi

    echo "Submitting job with $numSignalEncoderLayers numSignalEncoderLayers, $goldenRatio goldenRatio, $encodedDimension encodedDimension, $waveletType waveletType, $optimizer optimizer on $1"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSignalEncoderLayers_${numSignalEncoderLayers}_goldenRatio_${goldenRatio}_encodedDimension_${encodedDimension}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSignalEncoderLayers" "$goldenRatio" "$encodedDimension" "$1" "$waveletType" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
