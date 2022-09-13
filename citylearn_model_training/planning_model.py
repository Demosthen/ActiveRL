from typing import Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces.box import Box
import numpy as np
from model_utils import get_unit

class LitPlanningModel(pl.LightningModule):
    def __init__(self, obs_size: int = 1, act_size: int = 1, hidden_size: int = 1, lr: float = 0.0001, 
                    X_mean: Union[float, np.ndarray] = 0, X_std: Union[float, np.ndarray] = 1, y_mean: Union[float, np.ndarray] = 0, 
                    y_std: Union[float, np.ndarray] = 1, batch_norm: bool=True) -> None:
        super().__init__()
        self.lr = lr
        if None in [obs_size, act_size, hidden_size]:
            return
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.X_mean = torch.tensor(X_mean, device=self.device, dtype=torch.float16)
        self.X_std = torch.tensor(X_std, device=self.device, dtype=torch.float16)
        self.y_mean = torch.tensor(y_mean, device=self.device, dtype=torch.float16)
        self.y_std = torch.tensor(y_std, device=self.device, dtype=torch.float16)
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList([
            get_unit(obs_size + act_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            get_unit(hidden_size, hidden_size, batch_norm),
            nn.Linear(hidden_size, obs_size)
        ])
        self.save_hyperparameters()

    def preprocess(self, x):
        return (x - self.X_mean.to(self.device)) / self.X_std.to(self.device)

    def postprocess(self, y):
        return y * self.y_std.to(self.device) + self.y_mean.to(self.device)

    def from_np(self, x):
        return torch.tensor(x, device=self.device, dtype=torch.float32)

    def to_np(self, y):
        return y.detach().cpu().numpy()

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

    def forward_np(self, x):
        x = self.from_np(x)
        y = self.forward(x)
        return self.to_np(y)

    def eval_batchnorm(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.BatchNorm1d):
                        sublayer.eval()
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def shared_step(self, batch, batch_idx, return_mae=False):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if return_mae:
            mae_loss = torch.mean(torch.abs(y_hat - y))
            return loss, mae_loss
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.shared_step(train_batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, mae_loss = self.shared_step(val_batch, batch_idx, return_mae = True)
        self.log("val_loss", loss)
        self.log("val_mae", mae_loss)

    def get_reward(self, x, return_next_obs=False):
        self.eval_batchnorm()
        electricity_consumption_index = 23
        carbon_emission_index = 19
        electricity_pricing_index = 24
        next_obs = self.forward(x)
        net_electricity_consumption = next_obs[:, electricity_consumption_index]
        carbon_emission = torch.clip(next_obs[:, carbon_emission_index] * net_electricity_consumption, min=0)
        price = torch.clip(next_obs[:, electricity_pricing_index] * net_electricity_consumption, min=0)
        if return_next_obs:
            return torch.sum(carbon_emission + price), next_obs
        else:
            return torch.sum(carbon_emission + price)

    def compute_reward_uncertainty(self, obs, action, num_dropout_evals=10, return_avg_state=False):
        in_tensor = torch.concat([torch.tensor(obs, device=self.device), torch.tensor(action, device=self.device)], dim=-1)
        orig_mode = self.training
        self.train()
        rewards = []
        next_obss = []
        for _ in range(num_dropout_evals):
            rew, next_obs = self.get_reward(in_tensor, return_next_obs=True)
            rewards.append(rew)
            next_obss.append(next_obs)
        rewards = torch.stack(rewards)
        next_obss = torch.stack(next_obss)
        uncertainty = torch.std(rewards, dim=0)
        self.train(orig_mode)
        if return_avg_state:
            return uncertainty, torch.mean(next_obss, dim=0)
        else:
            return uncertainty

def get_planning_model(ckpt_file):
    if ckpt_file is None:
        return None
    model = LitPlanningModel.load_from_checkpoint(ckpt_file)
    return model
