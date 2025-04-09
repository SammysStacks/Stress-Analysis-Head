import torch.optim as optim
from torch.optim import Optimizer

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class optimizerMethods:

    def addOptimizer(self, submodel, model):
        # Get the model parameters.
        modelParams = [
            # Specify the profile parameters for the signal encoding.
            {'params': model.sharedSignalEncoderModel.healthGenerationModel.parameters(), 'weight_decay': modelConstants.userInputParams['physGenLR']/100, 'lr': modelConstants.userInputParams['physGenLR']},
            {'params': model.specificSignalEncoderModel.profileModel.parameters(), 'weight_decay': modelConstants.userInputParams['profileLR']/1000, 'lr': modelConstants.userInputParams['profileLR']},
            {'params': model.sharedSignalEncoderModel.fourierModel.parameters(), 'weight_decay': modelConstants.userInputParams['physGenLR']/100, 'lr': modelConstants.userInputParams['physGenLR']},

            # Specify the Lie manifold architecture parameters.
            # {'params': (param for name, param in model.named_parameters() if "givensRotationParams" in name), 'weight_decay': modelConstants.userInputParams['reversibleLR']/10, 'lr': modelConstants.userInputParams['reversibleLR']},
            {'params': (param for name, param in model.named_parameters() if "givensRotationParams" in name), 'weight_decay': modelConstants.userInputParams['reversibleLR'], 'lr': modelConstants.userInputParams['reversibleLR']*10},
            {'params': (param for name, param in model.named_parameters() if "basicEmotionWeights" in name), 'weight_decay': 5e-5, 'lr': 1e-4},
            {'params': (param for name, param in model.named_parameters() if "activationFunction" in name), 'weight_decay': 5e-5, 'lr': 1e-4},
            {'params': (param for name, param in model.named_parameters() if "jacobianParameter" in name), 'weight_decay': 5e-5, 'lr': 1e-4},
        ]

        # Set the optimizer and scheduler.
        optimizer = self.setOptimizer(modelParams, lr=1e-4, weight_decay=1e-5, optimizerType=modelConstants.userInputParams["optimizerType"])
        scheduler = self.getLearningRateScheduler(optimizer, submodel)

        return optimizer, scheduler

    def setOptimizer(self, params, lr, weight_decay, optimizerType):
        return self.getOptimizer(optimizerType=optimizerType, params=params, lr=lr, weight_decay=weight_decay, momentum=0.5)

    @staticmethod
    def getLearningRateScheduler(optimizer, submodel):
        # Options:
        # Slow ramp up: transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=30)
        # Cosine waveform: optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8, last_epoch=-1)
        # Reduce on plateau (need further editing of loop): optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # Defined lambda function: optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_function); lambda_function = lambda epoch: (epoch/50) if epoch < -1 else 1
        # torch.optim.lr_scheduler.constrainedLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1)
        return CosineAnnealingLR_customized(submodel=submodel, optimizer=optimizer,  T_max=1, absolute_min_lr=1e-5, multiplicativeFactor=2, numWarmupEpochs=modelConstants.numWarmupEpochs, warmupFactor=2, last_epoch=-1)

    @staticmethod
    def getOptimizer(optimizerType, params, lr, weight_decay, momentum=0.9):
        # General guidelines:
        #     Common WD values: 1E-2 to 1E-6
        #     Common LR values: 1E-6 to 1
        momentum_decay = modelConstants.userInputParams["momentum_decay"]
        beta1 = modelConstants.userInputParams["beta1"]
        beta2 = modelConstants.userInputParams["beta2"]

        if optimizerType == 'Adadelta':
            # Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
            # Use it when you don’t want to manually tune the learning rate.
            return optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=weight_decay)
        elif optimizerType == 'Adagrad':
            # Adagrad adapts the learning rates based on the parameters. It performs well with sparse data.
            # Use it if you are dealing with sparse features or in NLP tasks. Not compatible with GPU?!?
            return optim.Adagrad(params, lr=lr, lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0.2, eps=1e-10)
        elif optimizerType == 'Adam':
            # Adam is a first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates.
            # It's broadly used and suitable for most problems without much hyperparameter tuning.
            return optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=False, maximize=False)
        elif optimizerType == 'AdamW':
            # AdamW modifies the way Adam implements weight decay, decoupling it from the gradient updates, leading to a more effective use of L2 regularization.
            # Use when regularization is a priority and particularly when fine-tuning pre-trained models.
            return optim.AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=False, maximize=False)
        elif optimizerType == 'NAdam':
            # NAdam combines Adam with Nesterov momentum, aiming to combine the benefits of Nesterov and Adam.
            # Use in deep architectures where fine control over convergence is needed.
            return optim.NAdam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=True)
        elif optimizerType == 'RAdam':
            # RAdam (Rectified Adam) is an Adam variant that introduces a term to rectify the variance of the adaptive learning rate.
            # Use it when facing unstable or poor training results with Adam, especially in smaller sample sizes.
            return optim.RAdam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, decoupled_weight_decay=True)
        elif optimizerType == 'Adamax':
            # Adamax is a variant of Adam based on the infinity norm, proposed as a more stable alternative.
            # Suitable for embeddings and sparse gradients.
            return optim.Adamax(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        elif optimizerType == 'ASGD':
            # ASGD (Averaged Stochastic Gradient Descent) is used when you require robustness over a large number of epochs.
            # Suitable for larger-scale and less well-behaved problems; often used in place of SGD when training for a very long time.
            return optim.ASGD(params, lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=weight_decay)
        elif optimizerType == 'LBFGS':
            # LBFGS is an optimizer that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm, which is a quasi-Newton method.
            # Use it for small datasets where the exact second-order Hessian matrix computation is possible. Maybe cant use optimizer.step()??
            return optim.LBFGS(params, lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        elif optimizerType == 'RMSprop':
            # RMSprop is an adaptive learning rate method designed to solve Adagrad's radically diminishing learning rates.
            # It is well-suited to handle non-stationary data as in training neural networks.
            return optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay, momentum=momentum, centered=False)
        elif optimizerType == 'Rprop':
            # Rprop (Resilient Propagation) uses only the signs of the gradients, disregarding their magnitude.
            # Suitable for batch training, where the robustness of noisy gradients and the size of updates matters.
            return optim.Rprop(params, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif optimizerType == 'SGD':
            # SGD (Stochastic Gradient Descent) is simple yet effective, suitable for large datasets.
            # Use with momentum for non-convex optimization; ideal for most cases unless complexities require adaptive learning rates.
            return optim.SGD(params, lr=lr, momentum=momentum, dampening=0, weight_decay=weight_decay, nesterov=True)
        else: assert False, f"No optimizer initialized: {optimizerType}"


class CosineAnnealingLR_customized(optim.lr_scheduler.LRScheduler):
    def __init__(self, submodel: str, optimizer: Optimizer, T_max: int, absolute_min_lr: float, multiplicativeFactor: float, numWarmupEpochs: int, warmupFactor: float,  last_epoch: int = -1):
        self.multiplicativeFactor = multiplicativeFactor  # The multiplicative factor for the learning rate decay.
        self.absolute_min_lr = absolute_min_lr  # The absolute minimum learning rate to use.
        self.numWarmupEpochs = numWarmupEpochs  # The number of epochs to warm up the learning rate.
        self.warmupFactor = warmupFactor  # The factor to increase the learning rate during warmup.
        self.T_max = T_max  # The number of iterations before resetting the learning rate.
        self.warmupFlag = submodel == modelConstants.signalEncoderModel  # Flag to indicate if the warmup phase is active.
        self.numWarmupEpochs += 3 if submodel == modelConstants.emotionModel else 0

        # Call the parent class constructor
        super().__init__(optimizer, last_epoch)
        self.step()

    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""
        if self.last_epoch <= self.numWarmupEpochs and self.warmupFlag: return self.updateStep(multiplicativeFactor=self.warmupFactor, base_lrs=[max(self.absolute_min_lr, base_lr / (self.numWarmupEpochs - self.last_epoch + 1)) for base_lr in self.base_lrs])
        return self.updateStep(multiplicativeFactor=self.multiplicativeFactor, base_lrs=self.base_lrs)

    def updateStep(self, multiplicativeFactor, base_lrs):
        # Apply decay to each base learning rate
        decay_factor = multiplicativeFactor ** -((self.T_max - self.last_epoch) % self.T_max)
        return [max(self.absolute_min_lr, base_lr * decay_factor) for base_lr in base_lrs]
