import torch.nn as nn


class reversibleInterface(nn.Module):
    forwardDirection = True

    @classmethod
    def changeDirections(cls, newDirection):
        cls.forwardDirection = newDirection  # Modify class attribute
