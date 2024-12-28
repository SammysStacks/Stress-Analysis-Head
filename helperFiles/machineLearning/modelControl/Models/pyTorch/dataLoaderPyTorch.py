import keras
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration


class CustomDataset(Dataset):
    def __init__(self, features, labels, trainingMasks=None, testingMasks=None, variableSequence=False):
        # General
        self.trainingMasks = None
        self.testingMasks = None

        # Pad and realign the data if needed.
        if variableSequence: features = torch.tensor(keras.utils.pad_sequences(features, padding='post', dtype=float))

        # Read in the feature and labels.
        self.features = self.copyDataFormat(features)
        self.labels = self.copyDataFormat(labels)
        self.setMasks(trainingMasks, testingMasks)

    def setMasks(self, trainingMasks, testingMasks):
        if trainingMasks is not None:
            self.trainingMasks = self.copyDataFormat(trainingMasks).to(dtype=torch.bool)
            self.testingMasks = self.copyDataFormat(testingMasks).to(dtype=torch.bool)

            # Assert the integrity of the training/testing indices.
            if self.trainingMasks is None: assert self.testingMasks is None

    @staticmethod
    def copyDataFormat(data):
        if data is None: return None

        # Read in the feature labels.
        if isinstance(data, torch.Tensor): return data.clone()
        elif isinstance(data, (list, np.ndarray)): return torch.tensor(data)
        else: assert False, f"Unknown  instance type: {type(data)}"

    def getSignalInfo(self):
        return self.features.size()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get the feature and label
        feature = self.features[idx]
        label = self.labels[idx]

        # Return them if that is all.
        if self.testingMasks is None:
            return feature.clone(), label.clone()

        # Organize the training/testing masks.
        trainingMask = self.trainingMasks[idx]
        testingMask = self.testingMasks[idx]

        return feature.clone(), label.clone(), trainingMask.clone(), testingMask.clone()

    def getAll(self):
        if self.testingMasks is None:
            return self.features.clone(), self.labels.clone()

        return self.features.clone(), self.labels.clone(), self.trainingMasks.clone(), self.testingMasks.clone()

    def resetFeatures(self, features):
        self.features = features.clone()


class pytorchDataInterface:
    def __init__(self, batch_size=32, num_workers=0, shuffle=True, accelerator=None):
        # Specify the CPU or GPU capabilities.
        self.device = modelMigration.getModelDevice(accelerator)

        # General parameters
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pinMemory = False

        # CUDA differences.
        if self.device != "cpu":
            self.pinMemory = True

    def getDataLoader(self, allFeatures, allLabels, trainingMasks=None, testingMasks=None):
        dataset = CustomDataset(allFeatures, allLabels, trainingMasks, testingMasks, variableSequence=False)
        dataloader = DataLoader(
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pinMemory,
            persistent_workers=False,
            shuffle=self.shuffle,
            dataset=dataset,
        )

        return dataloader


if __name__ == "__main__":
    # Specify data params
    numDataPoints = 10
    numFeatures = 5
    # Initialize random data
    featureDataTest = torch.randn(numDataPoints, numFeatures)
    featureLabelsTest = torch.randn(numDataPoints, 1)

    # Organize the data into the pytorch format.
    pytorchDataClass = pytorchDataInterface(batch_size=32, num_workers=0, shuffle=True)
    trainingLoader = pytorchDataClass.getDataLoader(featureDataTest, featureLabelsTest)

    # Get all the data back.
    allInputsTest, allLabelsTest = trainingLoader.dataset.getAll()

    # For each batch of data points.
    for i, dataTest in enumerate(trainingLoader):
        # Extract the data and labels
        inputs, eachLabels = dataTest
