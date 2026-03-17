import torch.optim as optim

def build_scheduler(optimizer, epochs):
    """
    Builds the CosineAnnealingLR scheduler.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
