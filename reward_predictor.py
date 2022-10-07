import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import get_unit

class RewardPredictor(nn.Module):
    def __init__(self, in_size, hidden_size, batch_norm: bool=True, device="cpu") -> None:
        super().__init__()
        self.X_mean = nn.Parameter(torch.zeros(in_size), requires_grad=False)
        self.X_std = nn.Parameter(torch.ones(in_size), requires_grad=False)
        self.batch_norm = batch_norm
        self.y_mean = nn.Parameter(torch.zeros([1]), requires_grad=False)
        self.y_std = nn.Parameter(torch.ones([1]), requires_grad=False)
        self.momentum = 0.9
        self.layers = nn.ModuleList([
            get_unit(in_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            nn.Linear(hidden_size, 1)
        ])
        self.to(device)
        self.device = device

    def preprocess(self, x):
        ret = (x - self.X_mean.to(self.device)) / self.X_std.to(self.device)
        # Do not update on single samples
        if self.training and len(x) > 1:
            self.X_mean.data = self.momentum * self.X_mean + (1 - self.momentum) * torch.mean(x)
            self.X_std.data = self.momentum * self.X_std + (1 - self.momentum) * torch.std(x)
        return ret

    def postprocess(self, y):
        ret = y * self.y_std.to(self.device) + self.y_mean.to(self.device)
        # Do not update on single samples
        if self.training and len(y) > 1:
            self.y_mean.data = self.momentum * self.y_mean + (1 - self.momentum) * torch.mean(y)
            self.y_std.data = self.momentum * self.y_std + (1 - self.momentum) * torch.std(y)
        return ret
    
    def forward(self, x):
        x = self.preprocess(x)
        for i, layer in enumerate(self.layers):
            base = 0
            # Add residual connection if this is not
            # the first or last layer
            if i != 0 and i != len(self.layers) - 1:
                base = x
            x = layer(x) + base
        return self.postprocess(x)

    def eval_batchnorm(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.BatchNorm1d):
                        sublayer.eval()

    def compute_uncertainty(self, in_tensor, num_dropout_evals=10):
        orig_mode = self.training
        self.train()
        self.eval_batchnorm()
        rewards = []
        for _ in range(num_dropout_evals):
            rew = self.forward(in_tensor)
            rewards.append(rew)
        rewards = torch.stack(rewards)
        uncertainty = torch.mean(torch.var(rewards, axis=0))
        self.train(orig_mode)
        return uncertainty