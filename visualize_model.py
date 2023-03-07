import pickle 
import numpy as np
import matplotlib.pyplot as plt
import time
with open("checkpoints/reward_data2.pkl", "rb") as f:
    rews, bad_idxs = pickle.load(f)

def compute_rews(rews, bad_idxs):

    all_bad_idxs = set(bad_idxs["activerl"] + bad_idxs["vanilla"])
    # drop all bad indexes and sort by index
    activerl_dfs = rews["activerl"]
    idxs = ~activerl_dfs["idx"].isin(all_bad_idxs)
    activerl_dfs = activerl_dfs[idxs].sort_values("idx")
    vanilla_dfs = rews["vanilla"]
    vanilla_dfs = vanilla_dfs[idxs].sort_values("idx")
    xs = vanilla_dfs["x"]
    ys = vanilla_dfs["y"]
    return np.array(activerl_dfs["rew"]) , np.array(vanilla_dfs["rew"]), xs, ys

def compute_weighted_reward(active_rews, vanilla_rews, xs, ys, sigma):
    base_x = xs[0]
    base_y = ys[0]

    dists = (xs - base_x) ** 2 + (ys - base_y) ** 2
    likelihood = np.exp(-dists / (2 * sigma)) / sigma
    prob = likelihood / np.sum(likelihood)
    active_weighted_rew = active_rews @ prob
    vanilla_weighted_rew = vanilla_rews @ prob
    return active_weighted_rew, vanilla_weighted_rew

green = np.array([0,1,0])
red = np.array([1,0,0])
active_rews, vanilla_rews, xs, ys = compute_rews(rews, bad_idxs)
active_weighted_rews = []
vanilla_weighted_rews = []
sigmas = np.arange(0, 0.75, 0.01)
for sigma in sigmas:
    active_weighted_rew, vanilla_weighted_rew = compute_weighted_reward(active_rews, vanilla_rews, xs, ys, sigma)
    active_weighted_rews.append(active_weighted_rew)
    vanilla_weighted_rews.append(vanilla_weighted_rew)
plt.plot(sigmas, active_weighted_rews, label="ACTIVE RL")
plt.plot(sigmas, vanilla_weighted_rews, label = "VANILLA RL")
plt.xlabel("σ")
plt.ylabel("Robustness Metric")
plt.legend()
plt.tight_layout()
plt.savefig("weighted.png")
plt.figure()
print(f"ACTIVE WEIGHTED REW: {active_weighted_rew}, VANILLA WEIGHTED REW: {vanilla_weighted_rew}")
perf_comparisons = (active_rews > vanilla_rews)[:, None]
plt.scatter(xs[1:], ys[1:], c = perf_comparisons[1:] * green + (1-perf_comparisons)[1:] * red, s=10)
plt.scatter(xs[:1], ys[:1], c = perf_comparisons[:1] * green + (1-perf_comparisons)[:1] * red, marker="*", s=60, edgecolors="black")
plt.savefig("test.png")