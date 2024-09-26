#!/bin/bash
trainingData="2024-04-04 Final signalEncoder"
echo "Training data set to: $trainingData"

# Loop through encodedSamplingFreq values from 2 to 9
for encodedSamplingFreq in {2..15}; do
    # Loop through numEncodingLayers values from 0 to 10
    for numEncodingLayers in {0..15}; do
        # Construct the directory path for the first set of folders
        ##finalModelsDir="../../therapyHelperFiles/machineLearning/modelControl/_finalModels/emotionModel/metaTrainingModels/signalEncoder/$trainingData on CPU at encodedSamplingFreq ${encodedSamplingFreq} at numEncodingLayers ${numEncodingLayers}/"

        # Add, commit, and push changes for the first directory
        #git add "$finalModelsDir"
        #git commit -m "Added changes for signalEncoder with encodedSamplingFreq ${encodedSamplingFreq} and numEncodingLayers ${numEncodingLayers}"
        #git push

        # Construct the directory path for the second set of folders
        trainingFiguresDir="../../helperFiles/machineLearning/modelControl/Models/pyTorch/modelArchitectures/emotionModel/dataAnalysis/trainingFigures/signalEncoder/$trainingData on HPC-GPU at encodedSamplingFreq ${encodedSamplingFreq} at numEncodingLayers ${numEncodingLayers}/"

        # Add, commit, and push changes for the second directory
        git add "$trainingFiguresDir"
        git commit -m "Added changes for Training Figures signalEncoder with encodedSamplingFreq ${encodedSamplingFreq} and numEncodingLayers ${numEncodingLayers}"
        git push
    done
done

