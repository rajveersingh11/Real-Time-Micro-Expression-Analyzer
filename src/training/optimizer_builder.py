import torch.optim as optim

def build_optimizer(model, lr=1e-4, weight_decay=1e-5):
    """
    Builds the Adam optimizer.
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
