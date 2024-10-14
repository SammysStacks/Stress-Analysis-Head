#!/bin/bash

numLiftedChannelsList=(8 16 32 48)
# Total: 4

encodedSamplingFreqStart=6   # Minimum 4; Maximum 6
encodedSamplingFreqStep=-1
encodedSamplingFreqEnd=2     # Minimum: 2
# Total: 5

numEncodingLayersStart=8    # Absolute minimum is 0.
numEncodingLayersStep=-1
numEncodingLayersEnd=0      # Memory limited from 10-12.
# Total: 9

for numLiftedChannels in "${numLiftedChannelsList[@]}"
do
  for encodedSamplingFreq in $(seq $encodedSamplingFreqStart $encodedSamplingFreqStep $encodedSamplingFreqEnd)
  do
      for numEncodingLayers in $(seq $numEncodingLayersStart $numEncodingLayersStep $numEncodingLayersEnd)
      do
          echo "Submitting job with $numLiftedChannels numLiftedChannels and $encodedSamplingFreq encodedSamplingFreq and $numEncodingLayers numEncodingLayers on $1

          if [ "$1" == "CPU" ]; then
              sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${encodedSamplingFreq}_numEnc${numEncodingLayers}_$1" submitSignalEncoder_CPU.sh "$numLiftedChannels" "$numEncodingLayers" "$encodedSamplingFreq" "$1"
          elif [ "$1" == "GPU" ]; then
              sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${encodedSamplingFreq}_numEnc${numEncodingLayers}_$1" submitSignalEncoder_GPU.sh "$numLiftedChannels" "$numEncodingLayers" "$encodedSamplingFreq" "$1"
          else
              echo "No known device listed: $1"
          fi
      done
  done
done
