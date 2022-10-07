import sys
sys.path.append('./')
import pandas as pd
import numpy as np
num_episodes=12000
if __name__ == "__main__":
    # For some reason we have to define this store before importing the citylearnwrapper...
    data = pd.HDFStore("citylearn_model_training/planning_model_data.h5", 'r')
    new_store = pd.HDFStore("citylearn_model_training/planning_model_data_new.h5")
    num_rows = num_episodes * 8760
    obs_df = data.select("obs", stop=num_rows)
    action_df = data.select("actions", stop=num_rows)
    next_obs_df = data.select("next_obs", stop=num_rows)
    print(obs_df.shape)
    new_store.put("obs", obs_df)
    new_store.put("action", action_df)
    new_store.put("next_obs", next_obs_df)