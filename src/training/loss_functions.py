import torch.nn as nn

def build_loss(weight=None):
    """
    Builds the CrossEntropyLoss function.
    Args:
        weight (torch.Tensor, optional): A manual rescaling weight given to each class.
    """
    return nn.CrossEntropyLoss(weight=weight)
