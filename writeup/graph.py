import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os
from run_queries import *

def graph(graph_name, 
          TAG="ray/tune/evaluation/custom_metrics/reward_mean_mean", 
          FIGS_DIR = "figs/", 
          FORMAT='png', 
          window_width=5, 
          smooth_baseline=True, 
          out_file=None,
          data_func = lambda x: x,
          y_label="Reward",
          x_label="Timesteps",
          normalize=False,
          plot_legend=True,):
    plt.locator_params(axis='both', nbins=5)
    #xticks = [1000000 * i for i in range(5)]
    
    if out_file is None:
        out_file = graph_name
    
    linewidth=3
    axis_width=3
    def name_func(name):
        name = NAMES[graph_name][name]
        return name
    
    def color_func(name):
        colors = COLORS[graph_name]
        name_reverse_map = {v:k for k,v in NAMES[graph_name].items()}
        name = name_reverse_map[name]
        return colors[name]
        
    yticks = None#[-1.1 + 0.05 * i for i in range(1, 4)]
    offset = get_random_offset(f"{graph_name}.csv", TAG, random_name=BASELINE[graph_name], data_func=data_func) if normalize else 0
    print("OFFSET IS", offset)


    draw(f"{graph_name}.csv", 
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
        plot_legend=plot_legend,
        err_scale=1,
        reset=True,
        offset=offset,
        cutoff=3000000)

def graph_all(graph_name):
    avg_rew_tags = ["ray/tune/evaluation/custom_metrics/reward_mean_mean"]
    avg_power_tags = ["ray/tune/evaluation/custom_metrics/mean_power_penalty_mean"]
    avg_comfort_penalty_tags = ["ray/tune/evaluation/custom_metrics/mean_comfort_penalty_mean"]
    avg_comfort_violation_tags = ["ray/tune/evaluation/custom_metrics/comfort_violation_time____mean"]
    
    power_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_power_penalty_mean" for i in range(0, 6)]
    comfort_violation_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_comfort_violation_time____mean" for i in range(0, 6)]
    rew_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_reward_mean_mean" for i in range(0, 6)]
    comfort_penalty_tags = [f"ray/tune/evaluation/custom_metrics/env_{i}_mean_comfort_penalty_mean" for i in range(0, 6)]
    window_width = 5
    figs_dir = f"figs/{graph_name}"
    os.makedirs(figs_dir, exist_ok=True)

    for i, tag in enumerate(avg_rew_tags + rew_tags):
        out_file = tag.split("/")[-1]
        graph(graph_name, tag, out_file=out_file, window_width=window_width, plot_legend= i==1,
              y_label="Reward", FIGS_DIR=figs_dir)

    for i, tag in enumerate(avg_power_tags + power_tags):
        out_file = tag.split("/")[-1]
        data_func = lambda x: -x * 10000
        graph(graph_name, tag, out_file=out_file, window_width=window_width, plot_legend= i==1,
              y_label="Power (W)", data_func=data_func, normalize=False, FIGS_DIR=figs_dir)

    for i, tag in enumerate(avg_comfort_penalty_tags + comfort_penalty_tags):
        out_file = tag.split("/")[-1]
        data_func = lambda x: x * 10
        graph(graph_name, tag, out_file=out_file, window_width=window_width, plot_legend = i==1,
              y_label="Comfort (-Fanger PPD)", data_func=data_func, normalize=False, FIGS_DIR=figs_dir)

    for i, tag in enumerate(avg_comfort_violation_tags + comfort_violation_tags):
        out_file = tag.split("/")[-1]
        data_func = lambda x: x
        graph(graph_name, tag, out_file=out_file, window_width=window_width, plot_legend= i==1,
              y_label="% of Time\nASHRAE Comfort Violated", data_func=data_func, normalize=False, FIGS_DIR=figs_dir)

if __name__ == "__main__":
    graph_all("results")
    graph_all("vary_reg")
    graph_all("vary_lr")