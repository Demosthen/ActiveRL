import wandb
import pandas as pd 
import numpy as np
import sys
import run_queries
X_KEY = "ray/tune/counters/num_env_steps_sampled"
Y_KEYS = ["ray/tune/evaluation/custom_metrics/reward_mean_mean"] + \
        [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_power_penalty_mean" for i in range(0, 6)] + \
        [f"ray/tune/evaluation/custom_metrics/env_{i}_comfort_violation_time____mean" for i in range(0, 6)] + \
        [f"ray/tune/evaluation/custom_metrics/env_{i}_reward_mean_mean" for i in range(0, 6)] + \
        [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_comfort_penalty_mean" for i in range(0, 6)]
api = wandb.Api(timeout=20)
graph_name = sys.argv[1]
# Project is specified by <entity/project-name>
runs = api.runs("social-game-rl/active-rl",
    run_queries.QUERIES[graph_name]
    )
api.runs("social-game-rl/active-rl")
GROUP_BY = run_queries.GROUP_BY[graph_name]
vals = {}
cnts = {}
for run in runs:
    data_df = run.history(keys=Y_KEYS, samples=10000, x_axis=X_KEY)
    

    for tag in run.tags:
        if tag not in GROUP_BY:
            continue
        
        if tag not in vals:
            vals[tag] = data_df
            cnts[tag] = 1
        else:
            vals[tag] = pd.merge(vals[tag], data_df, on=X_KEY, how='outer', suffixes= [None, "_{}".format(cnts[tag])])
            cnts[tag] += 1
    
all_df = None
for tag, val in vals.items():
    curr_df = {X_KEY: val[X_KEY]}
    for Y_KEY in Y_KEYS:
        idxs = []
        for i in range(val.shape[1]):
            if Y_KEY + "_{}".format(i) in val:
                idxs.append(i)
        curr_vals = np.stack([val[Y_KEY]] + [val[Y_KEY + "_{}".format(i)] for i in idxs])
        curr_df.update({
                tag + ' - ' + Y_KEY : curr_vals.mean(axis=0), 
                tag + ' - ' + Y_KEY + '__std': curr_vals.std(axis=0), 
                tag + ' - ' + Y_KEY + '__ste': curr_vals.std(axis=0) / np.sqrt(len(curr_vals))})
    curr_df = pd.DataFrame.from_dict(curr_df)
    if all_df is None:
        all_df = curr_df
    else:
        all_df = pd.merge(all_df, curr_df, on=X_KEY, how='outer')
all_df = all_df.sort_values(by=X_KEY)


# all_df = pd.concat([name_df, config_df,summary_df], axis=1)
all_df.to_csv("data/{}.csv".format(graph_name), index=False)