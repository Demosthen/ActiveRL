import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os

def graph(file, 
          TAG="ray/tune/evaluation/custom_metrics/reward_mean_mean", 
          FIGS_DIR = "figs/", 
          FORMAT='png', 
          window_width=5, 
          smooth_baseline=True, 
          out_file=None,
          data_func = lambda x: x,
          y_label="Reward",
          x_label="Timesteps",
          normalize=True):
    plt.locator_params(axis='both', nbins=6)
    #xticks = [1000000 * i for i in range(5)]
    
    if out_file is None:
        out_file = file
    
    linewidth=3
    axis_width=3
    def name_func(name):
        if name == "results_random":
            name = "Random"
        elif name == "results_rbc":
            name = "RBC"
        elif name == "results_activeplr":
            name = "Active-PLR"
        elif name == "results_activerl":
            name = "Active-RL"
        elif name == "results_plr":
            name = "PLR"
        elif name == "results_random_reset":
            name = "Domain Randomization"
        elif name == "results_vanilla":
            name = "RL"
        return name
    
    def color_func(name):
        if name == "Random":
            return "red"
        elif name == "RBC":
            return "blue"
        elif name == "Active-PLR":
            return "green"
        elif name == "Active-RL":
            return "orange"
        elif name == "PLR":
            return "purple"
        elif name == "Domain Randomization":
            return "pink"
        elif name == "RL":
            return "black"
        
    yticks = None#[-1.1 + 0.05 * i for i in range(1, 4)]
    offset = get_random_offset(f"{file}.csv", TAG, random_name="results_random", data_func=data_func) if normalize else 0
    print("OFFSET IS", offset)


    draw(f"{file}.csv", 
        TAG, 
        x_label, 
        y_label, 
        fig_name=out_file,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        #xticks=xticks,
        yticks=yticks,
        # yticks=yticks,
        window_width=window_width, 
        data_func=data_func,
        color_func=color_func,
        smooth_baseline=smooth_baseline, 
        plot_legend=True,
        err_scale=1,
        reset=True,
        offset=offset,
        cutoff=3000000)


if __name__ == "__main__":
    avg_rew_tags = ["ray/tune/evaluation/custom_metrics/reward_mean_mean"]
    power_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_power_penalty_mean" for i in range(0, 6)]
    comfort_violation_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_comfort_violation_time____mean" for i in range(0, 6)]
    rew_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_reward_mean_mean" for i in range(0, 6)]
    comfort_penalty_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_comfort_penalty_mean" for i in range(0, 6)]
    window_width = 3

    for tag in avg_rew_tags + rew_tags:
        out_file = tag.split("/")[-1]
        graph("results", tag, out_file=out_file, window_width=window_width, y_label="Normalized Reward")

    for tag in power_tags:
        out_file = tag.split("/")[-1]
        data_func = lambda x: x * 10000
        graph("results", tag, out_file=out_file, window_width=window_width, 
              y_label="Power (W)", data_func=data_func, normalize=False)

    for tag in comfort_penalty_tags:
        out_file = tag.split("/")[-1]
        data_func = lambda x: x * 10
        graph("results", tag, out_file=out_file, window_width=window_width, 
              y_label="Comfort (Fanger PPD)", data_func=data_func, normalize=False)

    for tag in comfort_violation_tags:
        out_file = tag.split("/")[-1]
        data_func = lambda x: x
        graph("results", tag, out_file=out_file, window_width=window_width, 
              y_label="% of Time\nASHRAE Comfort Violated", data_func=data_func, normalize=False)