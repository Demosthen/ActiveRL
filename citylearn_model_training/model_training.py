# %%
import sys
 
# setting path
sys.path.append('../')
import pandas as pd
import numpy as np


# %%
data = pd.HDFStore("planning_model_data.h5", 'r')

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
# from citylearn.citylearn import CityLearnEnv
# from citylearn_wrapper import CityLearnEnvWrapper


import csv
from multiprocessing.spawn import prepare
from turtle import forward
from typing import Any

from planning_model import LitPlanningModel

# %%
hidden_size = 512
batch_norm = True
num_episodes = 50

# %%
# env = CityLearnEnvWrapper(env_config)
# obs_size = env.actionervation_space.shape[0]
# act_size = env.action_space.shape[0]


# %%

class PlanningDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        #self.filename = filename
        
        self.obs_df = data.get("obs")
        self.action_df = data.get("actions")
        self.next_obs_df = data.get("next_obs")
        data.close()
        #data = pd.read_csv(filename, delimiter="|").dropna()
        print("READ DATA ")
        # self.X = np.array([eval(data.iloc[i, 0]) + eval(data.iloc[i, 2]) for i in range(len(data))])
        print("READ X")
        # self.y = np.array([eval(data.iloc[i, 1]) for i in range(len(data))])
        print("READ Y")
        #del data
        print("DELETED DATA")
        obs_mean = np.mean(self.obs_df, axis=0)
        obs_std = np.std(self.obs_df, axis=0)
        action_mean = np.array(np.mean(self.action_df, axis=0))
        action_std = np.array(np.std(self.action_df, axis=0))
        next_obs_mean = np.array(np.mean(self.next_obs_df, axis=0))
        next_obs_std = np.array(np.std(self.next_obs_df, axis=0))
        self.X_mean = np.concatenate([obs_mean, action_mean])#np.mean(self.X, axis=0)
        self.X_std = np.concatenate([obs_std, action_std])#np.std(self.X, axis=0)
        print("X STATS")
        self.y_mean = next_obs_mean#np.mean(self.y, axis=0)
        self.y_std = next_obs_std#np.std(self.y, axis=0)
        print("Y STATS")

    def __len__(self):
        return len(self.obs_df)

    def __getitem__(self, index):
        obs = torch.tensor(self.obs_df.iloc[index], dtype=torch.float32)
        action = torch.tensor(self.action_df.iloc[index])
        next_obs = torch.tensor(self.next_obs_df.iloc[index], dtype=torch.float32)
        return torch.cat([obs, action]), next_obs
    

# %%
dataset = PlanningDataset()
obs_size = dataset.obs_df.shape[-1]
act_size = dataset.action_df.shape[-1]
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

#train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_base_idxs = np.array(list(range(8760 - 8760//5, 8760)))
train_base_idxs = np.array(list(range(8760 - 8760//5, 8760)))
val_idxs = [val_base_idxs + i * 8760 for i in range(len(dataset) // 8760)]
train_idxs = [train_base_idxs + i * 8760 for i in range(len(dataset) // 8760)]
val_idxs = np.concatenate(val_idxs)
train_idxs = np.concatenate(train_idxs)
train_set = torch.utils.data.Subset(dataset, train_idxs)
val_set = torch.utils.data.Subset(dataset, val_idxs)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=3)

# %%
print(obs_size, act_size)
model = LitPlanningModel(obs_size, act_size, hidden_size, num_layers=12, X_mean=dataset.X_mean, y_mean=dataset.y_mean, X_std=dataset.X_std, y_std=dataset.y_std)

# %%
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

wandb_logger = WandbLogger(project="active-rl-planning-model", entity="social-game-rl", log_model="all")
wandb_logger.experiment.config["exp_name"] = "alternate_validation_set_big_model_lr_scheduler"
checkpoint_callback = ModelCheckpoint(monitor="val_loss")
lr_callback = LearningRateMonitor(logging_interval="epoch")
trainer = pl.Trainer(gpus=1, precision=32, logger=wandb_logger, callbacks=[checkpoint_callback, lr_callback], auto_lr_find=False, max_epochs=1000)

#trainer.tune(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

# %%



