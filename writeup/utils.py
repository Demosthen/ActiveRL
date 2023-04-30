import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wandb
import writeup.run_queries as run_queries


def plot_scatter(start, rews, bad_idxs, base_key=None):
    green = np.array([0,1,0])
    red = np.array([1,0,0])
    all_bad_idxs = set(sum(bad_idxs.values(), []))
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    assert len(rews) == 2
    tags = list(rews.keys())
    tag1 = tags[0] if base_key is None else base_key
    tag2 = tags[1] if tags[0] == tag1 else tags[0]
    run_dfs = rews[tag1]
    print(f"{tag1}: ", run_dfs)
    idxs = ~run_dfs["idx"].isin(all_bad_idxs)
    run_dfs = run_dfs[idxs].sort_values("idx")
    compare_dfs = rews[tag2]
    idxs = ~compare_dfs["idx"].isin(all_bad_idxs)
    compare_dfs = compare_dfs[idxs].sort_values("idx")
    print(f"{tag2}: ", compare_dfs)

    

    rews = (np.array(run_dfs["rew"]) > np.array(compare_dfs["rew"]))[:, None]
    xs = compare_dfs["x"]
    ys = compare_dfs["y"]
    end = time.perf_counter()
    print(f"TOOK {end-start} SECONDS")

    plt.scatter(xs[1:], ys[1:], c = rews[1:] * green + (1-rews)[1:] * red, s=10)
    plt.scatter(xs[:1], ys[:1], c = rews[:1] * green + (1-rews)[:1] * red, marker="*", s=60, edgecolors="black")
    folder = "writeup/figs/comparisons/" 
    os.makedirs(folder, exist_ok=True)
    filename = folder + f"{tag1}_vs_{tag2}.png"
    plt.savefig(filename)

    wandb.log({"viz": wandb.Image(filename)})

def extract_rew_stats(run_dfs, all_bad_idxs, group=True):
    idxs = ~run_dfs["idx"].isin(all_bad_idxs)
    run_dfs = run_dfs[idxs].sort_values("idx")
    grouped_runs = run_dfs.groupby(["idx"]) if group else run_dfs
    cnt = grouped_runs.count()
    run_df_means = grouped_runs.mean()
    run_df_stes = grouped_runs.std() / np.sqrt(cnt) if np.all(grouped_runs.count() > 1) else pd.DataFrame(np.zeros_like(run_df_means), index=run_df_means.index, columns=run_df_means.columns)

    return run_df_means, run_df_stes


def plot_bars(start, rews, bad_idxs, graph_name, only_avg = False):
    plt.figure(figsize=(50, 10))
    plt.locator_params(axis='y', nbins=5)
    fontsize=26
    all_bad_idxs = set(sum(bad_idxs.values(), []))
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    width = 0.125
    num_tags = len(rews)
    colors = run_queries.COLORS[graph_name]
    labels = run_queries.NAMES[graph_name]
    all_avg_rew_means = []
    all_avg_rew_stes = []
    
    for i, tag in enumerate(rews.keys()):
        run_dfs = rews[tag]
        # print(f"{tag}: ", run_dfs)
        run_df_means, run_df_stes = extract_rew_stats(run_dfs, all_bad_idxs)
        all_run_df_means, all_run_df_stes = extract_rew_stats(run_dfs, all_bad_idxs, group=False)

        # rews = (np.array(run_df_means["rew"]) > np.array(compare_dfs["rew"]))[:, None]
        # xs = np.array(run_df_means.index)
        xs = np.concatenate([run_df_means.index, np.array([len(run_df_means.index)])]) if not only_avg else np.arange(2)
        # xs = np.array([0, 1])#len(run_df_means.index)])

        # print(xs)
        end = time.perf_counter()
        print(f"TOOK {end-start} SECONDS")

        color = colors[tag]
        label = labels[tag]

        rew_means = np.concatenate([run_df_means["rew"], np.array(all_run_df_means["rew"])[None]]) if not only_avg else np.concatenate([run_df_means["rew"][:1], np.array(all_run_df_means["rew"])[None]])
        rew_stes = np.concatenate([run_df_stes["rew"], np.array(all_run_df_stes["rew"])[None]]) if not only_avg else np.concatenate([run_df_stes["rew"][:1], np.array(all_run_df_stes["rew"])[None]])


        

        # Plots a bar chart with error bars with xs as the x-axis, comparing run_df_means and compare_df_means
        # with run_df_stes and compare_df_stes as the error bars
        # try:
        # print(rew_means.shape, rew_stes.shape, xs.shape, width, num_tags, i)
        
        rects = plt.bar(xs - width * num_tags / 2 + width * i, rew_means, yerr=rew_stes, width=width, align='center', alpha=0.5, ecolor='black', capsize=10, label=label, color=color)
        plt.bar_label(rects, padding=3, fmt="%.3f", fontsize=fontsize-10)
        # except Exception as e:
        #     breakpoint()

    
    # rects = plt.bar(num_tags - width * num_tags / 2 + width * i, all_run_df_means["rew"], yerr=all_run_df_stes["rew"], width=width, align='center', alpha=0.5, ecolor='black', capsize=10, label="Average", color=color)
    # plt.bar_label(rects, padding=3, fmt="%.3f", fontsize=fontsize)
    
    plt.legend(fontsize=fontsize+4)
    plt.ylabel("Average Reward", fontsize=fontsize)
    extreme_env_labels = ["Base", "Dry+Hot", "Wet+Windy", "Wet+Hot", "Dry+Cold", "Erratic", "Average"] if not only_avg else ["Base", "Average"]
    num_xticks = num_tags if not only_avg else 2
    plt.xticks(np.arange(num_xticks), labels=extreme_env_labels[:num_xticks], fontsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)


    folder = "writeup/figs/comparison_bars/" 
    os.makedirs(folder, exist_ok=True)
    filename = folder + f"{graph_name}.png"
    plt.savefig(filename)

    wandb.log({"viz": wandb.Image(filename)})