import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
class LitPlanningModel(pl.LightningModule):
    def __init__(self, obs_size = 1, act_size = 1, hidden_size = 1, lr = 0.0001, X_mean = 0, X_std = 1, y_mean = 0, y_std = 1, batch_norm=True) -> None:
        super().__init__()
        self.lr = lr
        if None in [obs_size, act_size, hidden_size]:
            return
        self.X_mean = torch.tensor(X_mean, device=self.device, dtype=torch.float16)
        self.X_std = torch.tensor(X_std, device=self.device, dtype=torch.float16)
        self.y_mean = torch.tensor(y_mean, device=self.device, dtype=torch.float16)
        self.y_std = torch.tensor(y_std, device=self.device, dtype=torch.float16)
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList([
            self.get_unit(obs_size + act_size, hidden_size),
            self.get_unit(hidden_size, hidden_size),
            self.get_unit(hidden_size, hidden_size),
            self.get_unit(hidden_size, hidden_size),
            nn.Linear(hidden_size, obs_size)
        ])
        self.save_hyperparameters()
    
    def get_unit(self, in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size), 
            nn.ReLU(),
            nn.BatchNorm1d(out_size) if self.batch_norm else nn.Identity(),
            nn.Dropout(),
            )

    def preprocess(self, x):
        return (x - self.X_mean.to(self.device)) / self.X_std.to(self.device)

    def postprocess(self, y):
        return y * self.y_std.to(self.device) + self.y_mean.to(self.device)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def shared_step(self, batch, batch_idx, return_mae=False):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if return_mae:
            eps = 0.00001
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

    def get_reward(self, x):
        self.eval()
        electricity_consumption_index = 23
        carbon_emission_index = 19
        electricity_pricing_index = 24
        next_obs = self.forward(x)
        net_electricity_consumption = next_obs[:, electricity_consumption_index]
        carbon_emission = torch.clip(next_obs[:, carbon_emission_index] * net_electricity_consumption, min=0)
        price = torch.clip(next_obs[:, electricity_pricing_index] * net_electricity_consumption, min=0)
        return torch.sum(carbon_emission + price)

    def compute_uncertainty(self, obs, action, num_dropout_evals=10):
        in_tensor = torch.concat([torch.tensor(obs, device=self.device), torch.tensor(action, device=self.device)], dim=-1)
        orig_mode = self.training
        self.train()
        rewards = []
        for _ in range(num_dropout_evals):
            rew = self.get_reward(in_tensor)
            rewards.append(rew)
        rewards = torch.stack(rewards)
        uncertainty = torch.std(rewards, dim=0)
        self.train(orig_mode)
        return uncertainty