import keras
# General
import numpy as np

# PyTorch
import torch
import tensorflow as tf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Import helper files
from .modelMigration import modelMigration


class CustomDataset(Dataset):
    def __init__(self, features, labels, trainingMasks=None, testingMasks=None, variableSequence=False):
        # General
        self.trainingMasks = None
        self.testingMasks = None

        # Pad and realign the data if needed.
        if variableSequence: features = torch.tensor(keras.utils.pad_sequences(features, padding='post', dtype=float))

        # Read in the feature and labels.
        self.labels = self.copyDataFormat(labels, dtype=torch.float)
        self.features = self.copyDataFormat(features, dtype=torch.float)
        self.setMasks(trainingMasks, testingMasks)

    def setMasks(self, trainingMasks, testingMasks):
        if trainingMasks is not None:
            self.trainingMasks = self.copyDataFormat(trainingMasks, dtype=torch.bool)
            self.testingMasks = self.copyDataFormat(testingMasks, dtype=torch.bool)

            # Assert the integrity of the training/testing indices.
            if self.trainingMasks is None: assert self.testingMasks is None

    @staticmethod
    def copyDataFormat(data, dtype=torch.float32):
        if data is None: return None

        # Read in the feature labels.
        if isinstance(data, torch.Tensor):
            return data.clone().detach().to(dtype)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data, dtype=dtype)
        else:
            assert False, f"Unknown  instance type: {type(data)}"

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
        self.device = modelMigration().getModelDevice(accelerator)
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

    def getDataLoader_variableLength(self, allFeatures, allLabels):
        dataset = CustomDataset(allFeatures, allLabels, variableSequence=True)
        dataloader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
        )
        return dataloader

    @staticmethod
    def collate_fn(batch):
        """ Custom collate function for padding variable-length sequences """
        # Separate data and labels from the batch
        data, labels = zip(*batch)

        # Pad the sequences. You can choose to pad with a specific value if needed (default is 0)
        padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)

        # Stack the labels into a single tensor
        labels = torch.stack(labels)

        return padded_data, labels

    @staticmethod
    def collate_fn2(loadedData):
        """ If given, called when iterating over the DataLoader """
        # Initialize holders
        batchLabels = []
        batchData = []

        # For each datapoint
        for dataInfo in loadedData:
            data, labels = dataInfo

            # Organize the data and labels
            batchLabels.append(labels)
            batchData.append(data)

        # Pad and realign the data
        paddedData = tf.keras.utils.pad_sequences(batchData, padding='pre', dtype=float)
        paddedData = torch.tensor(paddedData)

        return paddedData, torch.stack(batchLabels)


# -------------------------------------------------------------------------- #
# ------------------------------ User Testing ------------------------------ #

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
