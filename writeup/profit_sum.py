import pandas as pd
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
        # print(grp, tag)
        if TAG == tag.strip():
            groups[grp] = df.iloc[:, [0, i]]
            groups[grp] = groups[grp].dropna()
            groups[grp] = groups[grp][groups[grp].iloc[:, 0] < cutoff]
        if len(parts) > 1 and parts[0] == TAG:
            grp = name_func(grp)
            if parts[1] == 'MAX':
                group_maxs[grp] = df.iloc[:, [0, i]]
                group_maxs[grp] = group_maxs[grp].dropna()
                group_maxs[grp] = group_maxs[grp][group_maxs[grp].iloc[:, 0] < cutoff]
            elif parts[1] == 'MIN':
                group_mins[grp] = df.iloc[:, [0, i]]
                group_mins[grp] = group_mins[grp].dropna()
                group_mins[grp] = group_mins[grp][group_mins[grp].iloc[:, 0] < cutoff]
            elif parts[1] in ['std', 'ste']:
                print(orig_grp, grp)
                group_mins[grp] = df.iloc[:, [0, i]]
                group_mins[grp] = group_mins[grp].dropna()
                
                group_mins[grp] = group_mins[grp][group_mins[grp].iloc[:, 0] < cutoff]
                group_mins[grp].iloc[:, 1] = groups[orig_grp].iloc[:, 1] - group_mins[grp].iloc[:, 1]

                group_maxs[grp] = df.iloc[:, [0, i]]
                group_maxs[grp] = group_maxs[grp].dropna()
                group_maxs[grp] = group_maxs[grp][group_maxs[grp].iloc[:, 0] < cutoff]
                group_maxs[grp].iloc[:, 1] = groups[orig_grp].iloc[:, 1] + group_maxs[grp].iloc[:, 1]
                # print(groups[orig_grp].shape, group_maxs[grp].shape, group_mins[grp].shape)
            
    return groups, group_maxs, group_mins

def draw(PATH, TAG, data_dir="data/", days = [], latex = False, data_func=lambda x:str(x), name_func=lambda x:x, name_filter=lambda x: True, cutoff=1000000000):
    if data_dir:
        PATH = os.path.join(data_dir, PATH)
    groups, _, _ = read_results(PATH, TAG, name_func, cutoff)
    combined = list(groups.items())
    output = ""
    p_dict = {}     # there is likely a better way of doing this but this is a relatively easy fix
    name_list = []  # without rewriting code

    for col_name, df in combined:
        
        if not name_filter(col_name):
            #Skip this one
            continue
        #print(df)
        name = name_func(col_name)
        name_list.append(name)

        ys = df.iloc[:, 1].to_numpy()
        xs = df.iloc[:, 0].to_numpy()
        cum_ys = np.cumsum(ys)

        interval = xs[0]

        for d in days:
            assert d >= 0, "d must be nonnegative"
            assert d <= 10000, "10000 day maximum"
            idx = xs.size
            for i, val in enumerate(xs):
                if val > d:
                    idx = i
                    break
            p = 0
            if idx > 0:
                p = cum_ys[idx-1]
                p *= interval
                if idx != xs.size:
                    # this should only be equal if the size is exactly met at 10000
                    diff = d - xs[idx - 1]
                    p += diff * (ys[idx])
            else:
                p = ys[0] * d
            
            if latex:
                if name not in p_dict:
                    p_dict[name] = []
                p_dict[name].append(p)
                # output += " & " + data_func(p)
            else:
                if name == "Baseline":
                    print(name, "\t", d, "\t", p)
                else:
                    print(name, "\t\t", d, "\t", p)
    if latex:
        for i in range(len(days)):
            # for n in name_list:     # order is not guarenteed to be correct, instead will manually order
            output += " & " + data_func(p_dict[name_func("hnet")][i])
            output += " & " + data_func(p_dict[name_func("afl")][i])
            output += " & " + data_func(p_dict[name_func("baseline")][i])

    return output        



    
def make_table(files, days, latex=False, caption=None):
    TAG="ray/tune/custom_metrics/agg/reward_mean_mean"
    def data_func(num):
        return "{:.2f}".format(round(num / 100000, 2))
    def name_func(name):
        if "hnet" in name:
            return "PFH"
        elif "afl" in name:
            return "FedAvg"
        elif "baseline" in name:
            return "Baseline"
        else:
            return "ERROR"
    def filename_func(filename):
        words = filename.split("_")
        return words[0][0].upper() + words[0][1:] + ", " + str(int(words[1])) + " agents"

    if latex:
        print("\\begin{table}[h!]")
        print("\\centering")
        print("\\def\\arraystretch{1.5}")
        if len(days) > 1:
            output = "\\begin{tabular}{ | c "
            for _ in range(len(days)):
                output += "| c c c "
            output += "| }"
            print(output)
            print(" \\hline")
            output = " Scenario"
            for d in days:
                output += " & \\thead{{PFH \\\\{day}}} & \\thead{{FedAvg \\\\{day}}} & \\thead{{Baseline \\\\{day}}}".format(day = d)
            output +="\\\\ [0.5ex] "
            print(output)
        else:
            print("\\begin{tabular}{ | c | c c c | }")
            print(" \\hline")
            print(" Scenario & PFH & FedAvg & Baseline \\\\ [0.5ex] ")

        print(" \\hline")

    for f in files:
        if latex:
            output = " " + filename_func(f)
            output += draw(f + ".csv", 
                TAG, 
                data_func=data_func,
                name_func=name_func,
                days=days,
                latex=latex)
            output += " \\\\"
            print(output)
        else:
            draw(f + ".csv", 
                TAG,
                data_func=data_func,
                name_func=name_func,
                days=days,
                latex=latex)
    
    if latex:
        print(" \\hline")
        print("\\end{tabular}")
        print("\\medskip")
        if caption != None:
            print("\\caption{" + caption + "}")
        elif len(days) > 1:
            print("\\caption{Cumulative profits above base utility pricing.}")
        else:
            print("\\caption{Cumulative profits above base utility pricing after", days[0], "days.}")
        print("\\end{table}")


# make_table(["simple_05"], [100, 1000])
make_table(["simple_05", "simple_10", "simple_20", "medium_05", "medium_10",  "medium_20", "complex_05", "complex_10",
                "complex_20",], [2500, 10000], True, caption="Cumulative profits above base utility pricing, in hundred thousands.")
