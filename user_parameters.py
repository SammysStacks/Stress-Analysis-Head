
import argparse
import math

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants

parser = argparse.ArgumentParser(description='Specify model parameters')

# Add arguments for the general model
parser.add_argument('--submodel', type=str, default=modelConstants.signalEncoderModel, help='The component of the model we are training. Options: signalEncoderModel, emotionModel')
parser.add_argument('--optimizerType', type=str, default='NAdam', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc')
parser.add_argument('--learningProtocol', type=str, default='reversibleLieLayer', help='The learning protocol for the model: reversibleLieLayer')
parser.add_argument('--deviceListed', type=str, default="cpu", help='The device we are using: cpu, cuda')

# Add arguments for the health profile.
parser.add_argument('--initialProfileAmp', type=float, default=1e-3, help='The limits for profile initialization. Should be near zero')
parser.add_argument('--encodedDimension', type=int, default=256, help='The dimension of the health profile and all signals.')
parser.add_argument('--numProfileShots', type=int, default=32, help='The epochs for profile training: [16, 32]')

# Add arguments for the neural operator.
parser.add_argument('--waveletType', type=str, default='bior3.1', help='The wavelet type for the wavelet transform: bior3.1, bior3.3, bior2.2, bior3.5')
parser.add_argument('--operatorType', type=str, default='wavelet', help='The type of operator to use for the neural operator: wavelet')
parser.add_argument('--numIgnoredSharedHF', type=int, default=0, help='The number of ignored high frequency components: [0, 1, 2]')

# Add arguments for the signal encoder architecture.
parser.add_argument('--numSpecificEncoderLayers', type=int, default=1, help='The number of layers in the model: [1, 2]')
parser.add_argument('--numSharedEncoderLayers', type=int, default=7, help='The number of layers in the model: [2, 10]')

# Add arguments for observational learning.
parser.add_argument('--maxAngularThreshold', type=float, default=45, help='The larger rotational threshold in (degrees)')
parser.add_argument('--minAngularThreshold', type=float, default=5, help='The smaller rotational threshold in (degrees)')

# dd arguments for the emotion and activity architecture.
parser.add_argument('--numBasicEmotions', type=int, default=3, help='The number of basic emotions (basis states of emotions)')
parser.add_argument('--numActivityModelLayers', type=int, default=3, help='The number of layers in the activity model')
parser.add_argument('--numEmotionModelLayers', type=int, default=3, help='The number of layers in the emotion model')

# ----------------------- Training Parameters ----------------------- #

# Signal encoder learning rates.
parser.add_argument('--profileLR', type=float, default=0.01, help='The learning rate of the profile')
parser.add_argument('--physGenLR', type=float, default=4e-4, help='The learning rate of the profile generation (CNNs)')
parser.add_argument('--reversibleLR', type=float, default=0.05, help='The learning rate of the Lie manifold angles (degrees)')

# Add arguments for the emotion and activity architecture.
parser.add_argument('--momentum_decay', type=float, default=0.001, help='Momentum decay for the optimizer')
parser.add_argument('--beta1', type=float, default=0.7, help='Beta1 for the optimizer: 0.5 -> 0.99')  # 0.6, 0.7, 0.8
parser.add_argument('--beta2', type=float, default=0.8, help='Beta2 for the optimizer: 0.9 -> 0.999')  # 0.8, 0.9

# ----------------------- Compile Parameters ----------------------- #


def set_params():
    # Parse the arguments.
    userInputParams = vars(parser.parse_args())
    userInputParams['minWaveletDim'] = max(32, userInputParams['encodedDimension'] // (2 ** 4))
    userInputParams['minThresholdStep'] = userInputParams['reversibleLR']  # Keep as degrees
    userInputParams['reversibleLR'] = userInputParams['reversibleLR'] * math.pi / 180  # Keep as radians
    userInputParams['profileDimension'] = userInputParams['encodedDimension'] // 2  # The dimension of the profile.
    userInputParams['unifyModelWeights'] = True

    # Compie additional input parameters.
    userInputParams = modelParameters.getNeuralParameters(userInputParams)
    modelConstants.updateModelParams(userInputParams)
