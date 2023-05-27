from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

def ma(data, window_width=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    cumsum_vec = np.cumsum(data)
    data = np.array(data)
    # print("cs", cumsum_vec.shape)
    # print("d", data.shape)
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    # print("ma", ma_vec.shape)
    return np.concatenate([data[:window_width], ma_vec])

def read_results(PATH, TAG, name_func, cutoff=10000000):
    df = pd.read_csv(PATH)
    col_names = df.columns
    groups = {}
    group_mins = {}
    group_maxs = {}
    for i, name in enumerate(col_names):
        if i == 0:
            # Skip step parameter
            continue
        grp, tag = name.split(' - ')
        orig_grp = grp
        parts = tag.strip().split('__')
        print(grp, tag)
        if TAG == tag.strip():
            groups[grp] = df.iloc[:, [0, i]]
            groups[grp] = groups[grp]#.fillna(method="ffill")
            groups[grp] = groups[grp][groups[grp].iloc[:, 0] < cutoff]
        if len(parts) > 1 and parts[0] == TAG:
            grp = name_func(grp)
            if parts[1] == 'MAX':
                group_maxs[grp] = df.iloc[:, [0, i]]
                group_maxs[grp] = group_maxs[grp]#.fillna(method="ffill")
                group_maxs[grp] = group_maxs[grp][group_maxs[grp].iloc[:, 0] < cutoff]
            elif parts[1] == 'MIN':
                group_mins[grp] = df.iloc[:, [0, i]]
                group_mins[grp] = group_mins[grp]#.fillna(method="ffill")
                group_mins[grp] = group_mins[grp][group_mins[grp].iloc[:, 0] < cutoff]
            elif parts[1] in ['std', 'ste']:
                print(orig_grp, grp)
                group_mins[grp] = df.iloc[:, [0, i]]
                group_mins[grp] = group_mins[grp]#.fillna(method="ffill")
                
                group_mins[grp] = group_mins[grp][group_mins[grp].iloc[:, 0] < cutoff]
                group_mins[grp].iloc[:, 1] = groups[orig_grp].iloc[:, 1] - group_mins[grp].iloc[:, 1]

                group_maxs[grp] = df.iloc[:, [0, i]]
                group_maxs[grp] = group_maxs[grp]#.fillna(method="ffill")
                group_maxs[grp] = group_maxs[grp][group_maxs[grp].iloc[:, 0] < cutoff]
                group_maxs[grp].iloc[:, 1] = groups[orig_grp].iloc[:, 1] + group_maxs[grp].iloc[:, 1]
                print(groups[orig_grp].shape, group_maxs[grp].shape, group_mins[grp].shape)

    return groups, group_maxs, group_mins

def get_random_offset(PATH, TAG, data_dir="data/", name_func=lambda x:x, cutoff=1000000000, random_name="Random", data_func=lambda x: x):
    if data_dir:
        PATH = os.path.join(data_dir, PATH)
    groups, group_maxs, group_mins = read_results(PATH, TAG, name_func, cutoff)
    print(groups.keys())
    df = groups[random_name]
    ys = data_func(df.iloc[:, 1].dropna()).to_numpy()
    offset = ys.mean()

    return offset

def draw(PATH, TAG, x_label, y_label, data_dir="data/", yticks=None, xticks=None,linewidth=3, axis_width=3, format=".eps", figs_dir="figs/", name_func=lambda x:x, data_func=lambda x: x, fig_name="fig" , window_width=1, reset=True, name_filter=lambda x: True, color_func = None, smooth_baseline = False, err_scale=1, cutoff=1000000000, plot_legend=True, legend_font=16, include_errors=True, offset=0):
    """Draws a graph from a csv file with the given tag.
    The csv file should have a column named "Step" and a column for each tag.
    The tag should be in the format "Group - Tag", where Group is the name of the group of the data and Tag is the name of the data."""
    if data_dir:
        PATH = os.path.join(data_dir, PATH)
    groups, group_maxs, group_mins = read_results(PATH, TAG, name_func, cutoff)
    combined = list(groups.items())
    max_x = 0
    for col_name, df in combined:
        
        if not name_filter(col_name):
            #Skip this one
            continue
        #print(df)
        name = name_func(col_name)
        invalid_idxs = np.where(np.isnan(data_func(df.iloc[:, 1])))[0]
        max_idx = invalid_idxs.min() if len(invalid_idxs)>0 else len(df) # get first index that becomes nan

        ys = data_func(df.iloc[:, 1]).to_numpy()[:max_idx]
        xs = df.iloc[:, 0].to_numpy()[:max_idx]
        max_x = max(max_x, np.max(xs))
        if window_width > 1:
            ys = ma(ys, window_width)-offset
            # xs = xs[window_width//2:-window_width//2+1]
        
        
        errs = np.zeros_like(xs)
        if name in group_maxs:
            print(name, col_name, xs.shape, ys.shape, group_maxs[name].shape)
            max_df = group_maxs[name]
            max_ys = data_func(max_df.iloc[:, 1]).to_numpy()[:max_idx]
            max_xs = max_df.iloc[:, 0].to_numpy()[:max_idx]
            if window_width > 1 and name!="Random":
                max_ys = ma(max_ys, window_width)
                max_xs = max_xs[window_width//2:-window_width//2+1]
            min_df = group_mins[name]
            min_ys = data_func(min_df.iloc[:, 1]).to_numpy()[:max_idx]
            min_xs = min_df.iloc[:, 0].to_numpy()[:max_idx]
            errs = []
            for min_, max_ in zip(min_ys, max_ys):
                if max_ != min_:
                    errs.append((max_ - min_) / 2)
                elif len(errs) == 0:
                    errs.append(0)
                else:
                    errs.append(sum(errs[-5:])/len(errs[-5:]))
            #errs = errs[window_width//2:-window_width//2+1]
            if window_width > 1 and name != "Random":
                errs = ma(errs, window_width)
                errs /= err_scale
                #min_xs = min_xs[window_width//2:-window_width//2+1]
            #plt.fill_between(max_xs, min_ys, max_ys, alpha=0.2)
        if name == "Random":
            ys = ys.mean() * np.ones(ys.shape)
            errs = np.array(errs)
            errs = errs.mean() * np.ones(errs.shape) / np.sqrt(ys.shape[0])
        # breakpoint()
        if color_func == None:
            #plt.errorbar(xs, ys, label=name, linewidth=linewidth, yerr = errs)
            p = plt.plot(xs, ys, label=name, linewidth=linewidth)
            
        else:
            #plt.errorbar(xs, ys, label=name, linewidth=linewidth, c='b', yerr=errs)
            p = plt.plot(xs, ys, label=name, linewidth=linewidth, c=color_func(name))
        if include_errors:
            plt.fill_between(xs, (ys - errs)[::1], (ys + errs)[::1], alpha=0.3, interpolate=True, facecolor=p[-1].get_color())
    # if plot_norl:
    #     plt.plot(range(max_x), np.ones([max_x])*220, label="No RL", linewidth=linewidth, c="brown")


    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(axis_width)
    plt.gca().spines['bottom'].set_linewidth(axis_width)
    label_font=20
    tick_font=20
    plt.xlabel(x_label, fontsize=label_font)
    plt.ylabel(y_label, fontsize=label_font)
    if yticks is not None:
        plt.yticks(yticks, fontsize=tick_font)
    if xticks is not None:
        plt.xticks(xticks, fontsize=tick_font)

    plt.tick_params(axis='x', labelsize=tick_font)
    plt.tick_params(axis='y', labelsize=tick_font)

    # plt.ylim(150, 700)

    plt.tight_layout()
    save_fig_name = os.path.join(figs_dir, "{}.{}".format(fig_name, format))
    print(save_fig_name)
    plt.savefig(save_fig_name, format=format, dpi=180)

    if plot_legend:
        # save legend separately to a different file
        curr_ax = plt.gca()
        figlegend = plt.figure(figsize=(3,2))
        figlegend.legend(curr_ax.get_legend_handles_labels()[0], curr_ax.get_legend_handles_labels()[1])       
        legend_fig_name = os.path.join(figs_dir, "{}_legend.{}".format(fig_name, format))
        plt.tight_layout()
        plt.savefig(legend_fig_name, format=format, dpi=180)
        # plt.legend(fontsize=legend_font)

    if reset:
        plt.clf()
        plt.figure()