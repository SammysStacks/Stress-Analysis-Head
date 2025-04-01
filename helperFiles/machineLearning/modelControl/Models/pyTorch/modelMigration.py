import os

import torch
import torch.nn as nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters


class modelMigration:

    def __init__(self, accelerator=None, debugFlag=False):
        # Create folders to save the data in.
        self.saveModelFolder = os.path.normpath(os.path.dirname(__file__) + "/../../../_finalModels/") + "/"
        self.saveModelFolder = os.path.relpath(os.path.normpath(self.saveModelFolder), os.getcwd()) + "/"
        os.makedirs(self.saveModelFolder, exist_ok=True)  # Create the folders if they do not exist.

        # Specify the accelerator parameters.
        self.device = self.getModelDevice(accelerator)
        self.accelerator = accelerator
        self.debugFlag = debugFlag

        # Model identifiers.
        self.classAttributeKey = "classAttributes"
        self.classAttributeString = " Attributes"
        self.sharedWeightsName = "sharedData"
        self.modelStateKey = "modelState"

    def replaceFinalModelFolder(self, folderPath):
        self.saveModelFolder = "".join([self.saveModelFolder.split("_finalModels")[0], folderPath])

    # ---------------------------------------------------------------------- #
    # ------------------------- Specify the Device ------------------------- #

    @staticmethod
    def getModelDevice(accelerator=None):
        if accelerator: return accelerator.device
        else: return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------- #
    # ------------------- Alter/Transfer Model Parameters ------------------ #

    def copyModelWeights(self, modelClass, modelWeights):
        layerInfo = {}
        # For each parameter in the model
        for layerName, layerParams in modelClass.model.named_parameters():
            currentSubmodel = self.getSubmodel(layerName)

            # If the layer should be saved.
            if currentSubmodel in modelWeights:
                # Save the layer (bias and weight individually)
                layerInfo[layerName] = layerParams.data.clone()

        return layerInfo

    def unifyModelWeights(self, allModels, modelWeights, layerInfo):
        if layerInfo is None: return None

        # For each model provided.
        for modelInd in range(len(allModels)):
            pytorchModel = allModels[modelInd].model

            # For each parameter in the model
            for layerName, layerParams in pytorchModel.named_parameters():
                currentSubmodel = self.getSubmodel(layerName)

                # If the layer should be saved.
                if currentSubmodel in modelWeights:
                    assert layerName in layerInfo, layerName
                    layerParams.data = layerInfo[layerName].clone()

    @staticmethod
    def getSubmodel(layerName):
        modelBlocks = [modelName for modelName in layerName.split(".") if "Model" in modelName]
        if len(modelBlocks) == 0: return layerName
        else: return modelBlocks[0]

    # ---------------------------------------------------------------------- #
    # ----------------------- Reset Model Parameters ----------------------- #

    @staticmethod
    def resetLayerStatistics(allModels):
        # For each model provided.
        for modelInd in range(len(allModels)):
            pytorchModel = allModels[modelInd].model

            for module in pytorchModel.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    module.reset_running_stats()  # This will reset the running mean and variance

    # ---------------------------------------------------------------------- #
    # ----------------- Common Model Saving/Loading Methods ---------------- #

    @staticmethod
    def _getAllModelChildren(model):
        # Iterate over all submodules using named_children() to include their names
        for name, submodule in model.named_children():
            print(name)

    def _filterStateDict(self, model, sharedModelWeights, submodelsSaving):
        unwrapped_model = self.accelerator.unwrap_model(model)

        # Initialize dictionaries to hold shared and specific parameters
        datasetSpecific_params = {}
        shared_params = {}

        # Iterate over all the parameters in the model's state_dict
        for name, param in unwrapped_model.state_dict().items():
            if name.startswith("module."): name = name[len("module."):]
            currentSubmodel = self.getSubmodel(name)

            # Check if these weights are a part of a model we are saving.
            if not any(currentSubmodel.startswith(submodel) for submodel in submodelsSaving):
                continue

            # Check if the parameter name starts with any of the prefixes in sharedModelWeights
            if any(currentSubmodel.startswith(shared) for shared in sharedModelWeights):
                # If the parameter is shared, add it to the shared_params dictionary
                shared_params[name] = param
            else:
                # If the parameter is not shared, add it to the specific_params dictionary
                datasetSpecific_params[name] = param

        return shared_params, datasetSpecific_params

    def _filterClassAttributes(self, model, sharedModelWeights, submodelsSaving):
        unwrapped_model = self.accelerator.unwrap_model(model)

        # Initialize dictionaries to hold shared and specific attributes
        shared_attributes = {}
        dataset_specific_attributes = {}

        # Iterate over all submodules using named_children() to include their names
        for name, submodule in unwrapped_model.named_children():
            if name.startswith("module."): name = name[len("module."):]
            currentSubmodel = self.getSubmodel(name)

            # Check if these weights are a part of a model we are saving.
            if not any(currentSubmodel.startswith(submodel) for submodel in submodelsSaving):
                continue

            # Check if the submodule name starts with any of the prefixes in sharedModelWeights
            if any(currentSubmodel.startswith(submodel) for submodel in sharedModelWeights):
                # If the submodule is shared, add its attributes to the shared_attributes dictionary
                shared_attributes[name] = self._removeBadAttributes(submodule.__dict__)
            else:
                # If the submodule is not shared, add its attributes to the specific_attributes dictionary
                dataset_specific_attributes[name] = self._removeBadAttributes(submodule.__dict__)

        assert shared_attributes == {}, "Attributes are never shared"
        return shared_attributes, dataset_specific_attributes

    @staticmethod
    def _removeBadAttributes(modelAttributes):
        newModelAttributes = {}

        # Iterate over all attributes in the model
        for attr_name, attr_value in modelAttributes.items():
            # Remove hidden attributes.
            if attr_name.startswith(("_", "accelerator")): continue
            if 'accelerator' in attr_name: continue
            if 'accelerate' in attr_name: continue

            # Remove the DDP addon for modules.
            if attr_name.startswith("module."): attr_name = attr_name[len("module."):]

            newModelAttributes[attr_name] = attr_value

        return newModelAttributes

    def _compileModelBaseName(self, submodel, datasetName, trainingDate, numEpochs):
        # Organize information about the model.
        trainingModelName = modelParameters.embedInformation(submodel, trainingDate)

        # Compile the location to save/load the model.
        modelFolderPath = self.saveModelFolder + f"{submodel}/{trainingModelName}/{datasetName}/"

        if numEpochs == -1 and os.path.exists(modelFolderPath):
            # List all folders in the model folder path.
            epochsFolders = [d for d in os.listdir(modelFolderPath) if os.path.isdir(os.path.join(modelFolderPath, d)) and d.startswith('Epoch ')]
            # Extract epoch numbers and find the highest one.
            epochNumbers = [int(folder.split(' ')[1]) for folder in epochsFolders]
            highestEpoch = max(epochNumbers)
            numEpochs = highestEpoch  # Use the highest epoch number.

        # Update modelFilePath based on potentially updated numEpochs
        modelFilePath = modelFolderPath + f"Epoch {numEpochs}/"

        # Compile the filename to save/load the model.
        modelBaseName = modelFilePath + f"{trainingModelName} {datasetName} {submodel} at {numEpochs} Epochs"

        return modelBaseName

    @staticmethod
    def _createFolder(filePath):
        os.makedirs(os.path.dirname(filePath), exist_ok=True)

    # ------------------------ Saving Model Methods ------------------------ #

    def saveModels(self, modelPipelines, datasetNames, sharedModelWeights, submodelsSaving, submodel, trainingDate, numEpochs, saveModelAttributes=True):
        # Assert the integrity of the input variables.
        assert len(modelPipelines) == len(datasetNames), f"You provided {len(modelPipelines)} models to save, but only {len(datasetNames)} datasetNames."
        assert 0 < len(modelPipelines), "No models provided to save."
        subAttributesSaving = submodelsSaving

        # For each model, save the shared and specific weights
        for datasetInd, (modelPipeline, datasetName) in enumerate(zip(modelPipelines, datasetNames)):
            self._saveModel(modelPipeline.model, datasetName, sharedModelWeights, submodelsSaving, subAttributesSaving, submodel, trainingDate, numEpochs, saveModelAttributes, datasetInd)

    def _saveModel(self, model, datasetName, sharedModelWeights, submodelsSaving, subAttributesSaving, submodel, trainingDate, numEpochs, saveModelAttributes=True, datasetInd=0):
        # Create a path to where we want to save the model.
        modelBaseName = self._compileModelBaseName(submodel, datasetName, trainingDate, numEpochs)
        sharedModelBaseName = self._compileModelBaseName(submodel, self.sharedWeightsName, trainingDate, numEpochs)

        # Filter the state_dict based on sharedModelWeights
        shared_params, specific_params = self._filterStateDict(model, sharedModelWeights, submodelsSaving)
        shared_attributes, specific_attributes = self._filterClassAttributes(model, sharedModelWeights=[], submodelsSaving=subAttributesSaving)

        # Save the pytorch models.
        self._savePyTorchModel(specific_params, specific_attributes, modelBaseName, saveModelAttributes)  # Save dataset-specific parameters
        if datasetInd == 0: self._savePyTorchModel(shared_params, shared_attributes, sharedModelBaseName, saveModelAttributes)  # Save shared parameters

    def _savePyTorchModel(self, modelParams, attributes, modelBaseName, saveModelAttributes):
        # Only save the file if there is data to save.
        if not modelParams and not attributes: return None

        # Prepare to save the model
        saveModelPath = modelBaseName + ".pth"

        # Create the folders if they do not exist.
        self._createFolder(modelBaseName)

        # Combine saving information.
        model_info = {self.modelStateKey: modelParams}
        if saveModelAttributes: model_info[self.classAttributeKey] = attributes

        # Save the model information.
        self.accelerator.save(model_info, saveModelPath)

    # ------------------------ Loading Model Methods ----------------------- #
    
    def loadModels(self, allModelPipelines, submodel, trainingDate, numEpochs, metaTraining, loadModelAttributes=True, loadModelWeights=True):
        print("\nLoading in previous weights and attributes")

        # Iterate over each model.
        for modelPipeline in allModelPipelines:
            self._loadModel(modelPipeline.model, modelPipeline.datasetName, submodel, trainingDate, numEpochs, metaTraining, loadModelAttributes, loadModelWeights)

    def _loadModel(self, model, datasetName, submodel, trainingDate, numEpochs, metaTraining, loadModelAttributes=True, loadModelWeights=True):
        # Construct base names for loading the model and attributes
        modelBaseName = self._compileModelBaseName(submodel, datasetName, trainingDate, numEpochs)
        sharedModelBaseName = self._compileModelBaseName(submodel, self.sharedWeightsName, trainingDate, numEpochs)

        # Load in the pytorch models.
        self._loadPyTorchModel(model, modelBaseName, loadModelAttributes, loadModelWeights, submodel)  # Load dataset-specific parameters
        self._loadPyTorchModel(model, sharedModelBaseName, loadModelAttributes, loadModelWeights, submodel)  # Save shared parameters

    def _loadPyTorchModel(self, model, modelBaseName, loadModelAttributes, loadModelWeights, submodel):
        # Prepare to save the attributes
        loadModelPath = modelBaseName + ".pth"

        # If the model exists.
        if model and os.path.exists(loadModelPath):
            model.eval()  # Set the model to evaluation mode

            # Load the model information.
            loadedModelInfo = torch.load(loadModelPath, map_location=self.device, weights_only=False)

            # Check if the model is wrapped with DDP
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # Adjusting the model state dictionary
                model_state_dict = loadedModelInfo[self.modelStateKey]
                new_model_state_dict = {"module." + key: value for key, value in model_state_dict.items()}
                loadedModelInfo[self.modelStateKey] = new_model_state_dict

                # Adjusting the class attribute dictionary
                class_attribute_dict = loadedModelInfo[self.classAttributeKey]
                new_class_attribute_dict = {"module." + key: value for key, value in class_attribute_dict.items()}
                loadedModelInfo[self.classAttributeKey] = new_class_attribute_dict

            if loadModelWeights:
                # Load in the model parameters.
                model.load_state_dict(loadedModelInfo[self.modelStateKey], strict=False)  # strict=False to allow for loading only matching parts
                model.to(self.device)

            if loadModelAttributes:
                # Load in the model attributes.
                self._loadModelAttributes(model, loadedModelInfo)
        else:
            modelNotLoaded = modelBaseName.split(f"/{submodel}/")[-1].split("/")[1]
            if self.debugFlag: print(f"\tNot loading the {submodel} model for {modelNotLoaded} data")

    def _loadModelAttributes(self, model, loadedModelInfo):
        modelAttributes = loadedModelInfo[self.classAttributeKey]

        # Iterate over all submodules using named_children() to include their names
        for name, submodule in model.named_children():
            # Check if the submodule name starts with any of the prefixes in sharedModelWeights
            if name in modelAttributes:
                submodule.__dict__.update(modelAttributes[name])

    # ---------------------------------------------------------------------- #                    
# -------------------------------------------------------------------------- #
