import torch.nn as nn

def get_unit(in_size, out_size, batch_norm=True):

    return nn.Sequential(
        nn.Linear(in_size, out_size), 
        nn.ReLU(),
        nn.BatchNorm1d(out_size) if batch_norm else nn.Identity(),
        nn.Dropout(),
        )