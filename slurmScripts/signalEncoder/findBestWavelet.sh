#!/bin/bash

waveletTypes=('cgau1' 'cgau2' 'cgau3' 'cgau4' 'cgau5' 'cgau6' 'cgau7' 'cgau8' 'smor' 'shan' 'gaus1' 'gaus2' 'gaus3' 'gaus4' 'gaus5' 'gaus6' 'gaus7' 'gaus8' 'morl' 'mexh' 'fbsp')
waveletTypes=('dmey' 'db31' 'db32' 'db33' 'db34' 'db35' 'db36' 'db37' 'db38' 'db24' 'db25' 'db26' 'db27' 'db28' 'db29' 'db30' 'coif15' 'coif16' 'coif17' 'coif8' 'coif9' 'coif10' 'coif11' 'coif12' 'coif13' 'coif14')

waveletTypes=( \
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

    # 17 coif wavelets
    'coif1' 'coif2' 'coif3' 'coif4' 'coif5' 'coif6' 'coif7' 'coif8' 'coif9' 'coif10' \
    'coif11' 'coif12' 'coif13' 'coif14' 'coif15' 'coif16' 'coif17' \
)

numSignalEncoderLayers=16
encodedDimension=256
optimizer='AdamW'
goldenRatio=2

for waveletType in "${waveletTypes[@]}"
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
