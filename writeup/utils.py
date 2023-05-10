import os
import time
from matplotlib import lines, patches, pyplot as plt
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

    name1 = run_queries.NAMES["sim2real"][tag1]
    name2 = run_queries.NAMES["sim2real"][tag2]
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

    red_patch = patches.Patch(color='red', label=f'{name1} underperforms {name2}')
    green_patch = patches.Patch(color='green', label=f'{name1} outperforms {name2}')
    star = lines.Line2D([], [], markeredgecolor='black', color='white', marker='*', linestyle='None',
                          markersize=10, label='Base environment')
    plt.legend(handles=[green_patch, red_patch, star])
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")


    folder = "writeup/figs/comparisons/" 
    os.makedirs(folder, exist_ok=True)
    filename = folder + f"{tag1}_vs_{tag2}.png"
    plt.savefig(filename)

    wandb.log({"viz": wandb.Image(filename)})

def plot_boxplot(start, rews, bad_idxs, base_key=None):
    green = np.array([0,1,0])
    red = np.array([1,0,0])
    all_bad_idxs = set(sum(bad_idxs.values(), []))
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    assert len(rews) == 2
    tags = list(rews.keys())
    tag1 = tags[0] if base_key is None else base_key
    tag2 = tags[1] if tags[0] == tag1 else tags[0]

    name1 = run_queries.NAMES["sim2real"][tag1]
    name2 = run_queries.NAMES["sim2real"][tag2]
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

    red_patch = patches.Patch(color='red', label=f'{name1} underperforms {name2}')
    green_patch = patches.Patch(color='green', label=f'{name1} outperforms {name2}')
    star = lines.Line2D([], [], markeredgecolor='black', color='white', marker='*', linestyle='None',
                          markersize=10, label='Base environment')
    plt.legend(handles=[green_patch, red_patch, star])
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")


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


def plot_bars(start, rews, bad_idxs, graph_name, only_avg = False, relative=False):
    plt.figure(figsize=(25, 10))
    plt.locator_params(axis='y', nbins=5)
    fontsize=26
    all_bad_idxs = set(sum(bad_idxs.values(), []))
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    width = 0.1 if not relative else 2
    num_tags = len(rews)
    colors = run_queries.COLORS[graph_name]
    labels = run_queries.NAMES[graph_name]
    all_avg_rew_means = []
    all_avg_rew_stes = []

    BASELINES = {
        "results_plr": -1.035,
        "results_rbc": -1.028,
        "results_random_reset": -1.049,
        "results_random": -1.078,
        "results_activeplr": -0.9591,
        "results_activerl": -0.9868,
        "results_vanilla": -1.036,
        "results_robust_plr": -1.034,
        "results_robust_grounded_plr": -0.988
    }
    
    for i, tag in enumerate(rews.keys()):
        run_dfs = rews[tag]
        # print(f"{tag}: ", run_dfs)
        run_df_means, run_df_stes = extract_rew_stats(run_dfs, all_bad_idxs)
        all_run_df_means, all_run_df_stes = extract_rew_stats(run_dfs, all_bad_idxs, group=False)

        # rews = (np.array(run_df_means["rew"]) > np.array(compare_dfs["rew"]))[:, None]
        # xs = np.array(run_df_means.index)
        if relative:
            xs = np.arange(1)
        elif only_avg:
            xs = np.arange(2)
        else:
            xs = np.concatenate([run_df_means.index, np.array([len(run_df_means.index)])])

        # print(xs)
        end = time.perf_counter()
        print(f"TOOK {end-start} SECONDS")

        color = colors[tag]
        label = labels[tag]

        avg_rew_mean = run_df_means["rew"].mean(axis=0)
        avg_rew_stes = ((run_df_stes["rew"] ** 2).mean())**0.5

        if relative:
            rew_means = 100*np.array(avg_rew_mean - BASELINES[tag]) / np.abs(BASELINES[tag])
            rew_stes = 100*np.array(avg_rew_stes) / np.abs(BASELINES[tag])
        elif only_avg:
            rew_means = np.concatenate([run_df_means["rew"][:1], avg_rew_mean[None]])
            rew_stes = np.concatenate([run_df_stes["rew"][:1], avg_rew_stes[None]])
        else:
            rew_means = np.concatenate([run_df_means["rew"], avg_rew_mean[None]])
            rew_stes = np.concatenate([run_df_stes["rew"], avg_rew_stes[None]])
        # rew_means = np.concatenate([run_df_means["rew"], avg_rew_mean[None]]) if not only_avg else np.concatenate([run_df_means["rew"][:1], avg_rew_mean[None]])
        # rew_stes = np.concatenate([run_df_stes["rew"], avg_rew_stes[None]]) if not only_avg else np.concatenate([run_df_stes["rew"][:1], avg_rew_stes[None]])

        # rew_means = np.concatenate([run_df_means["rew"], np.array(all_run_df_means["rew"])[None]]) if not only_avg else np.concatenate([run_df_means["rew"][:1], np.array(all_run_df_means["rew"])[None]])
        # rew_stes = np.concatenate([run_df_stes["rew"], np.array(all_run_df_stes["rew"])[None]]) if not only_avg else np.concatenate([run_df_stes["rew"][:1], np.array(all_run_df_stes["rew"])[None]])
       

        # Plots a bar chart with error bars with xs as the x-axis, comparing run_df_means and compare_df_means
        # with run_df_stes and compare_df_stes as the error bars
        # try:
        # print(rew_means.shape, rew_stes.shape, xs.shape, width, num_tags, i)
        padding = 1 if relative else 0
        total_width = width + padding
        rects = plt.bar(xs - total_width * num_tags / 2 + total_width * i, rew_means, yerr=rew_stes, width=width, align='center', alpha=0.5, ecolor='black', capsize=10, label=label, color=color)
        if only_avg or relative:
            bar_label_fontsize = fontsize - 14 if only_avg else fontsize - 6
            plt.bar_label(rects, padding=3, fmt="%.3f", fontsize=bar_label_fontsize)
        # except Exception as e:
        #     breakpoint()

    
    # rects = plt.bar(num_tags - width * num_tags / 2 + width * i, all_run_df_means["rew"], yerr=all_run_df_stes["rew"], width=width, align='center', alpha=0.5, ecolor='black', capsize=10, label="Average", color=color)
    # plt.bar_label(rects, padding=3, fmt="%.3f", fontsize=fontsize)
    if not relative:
        plt.legend(fontsize=fontsize)
    ylabel = "Relative Performance\nDegradation (%)" if relative else "Average Reward"
    plt.ylabel(ylabel, fontsize=fontsize)
    extreme_env_labels = ["Base", "Dry+Hot", "Wet+Windy", "Wet+Hot", "Dry+Cold", "Erratic", "Average"] if not only_avg else ["Base", "Average"]
    
    if relative:
        xticks = xs - total_width * num_tags / 2 + total_width * np.arange(num_tags)
        print(xticks, xs)
        xlabels = []
        for tag in rews.keys():
            label = labels[tag]
            if label =="Domain Randomization":
                label = "Domain\nRandomization"
            xlabels.append(label)
        plt.xticks(xticks, labels=xlabels, fontsize=fontsize)
    else:
        num_xticks = len(extreme_env_labels) if not only_avg else 2
        plt.xticks(np.arange(num_xticks), labels=extreme_env_labels[:num_xticks], fontsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)


    folder = "writeup/figs/comparison_bars/" 
    os.makedirs(folder, exist_ok=True)
    prefix = "relative_" if relative else ""
    filename = folder + f"{prefix}{graph_name}.png"
    plt.tight_layout()
    plt.savefig(filename)

    wandb.log({"viz": wandb.Image(filename)})