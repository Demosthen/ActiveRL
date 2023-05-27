QUERIES = {
            "results": {"$or": [
                        {"tags": {"$in": ["results"]}},
            ]},
            "debug": {"$or": [
                        {"tags": {"$in": ["results"]}},
            ]},
            "sim2real": {"$or": [
                        {"tags": {"$in": ["median_random", "median_activeplr", "median_activerl", "median_robust_plr", "median_random_reset", "median_vanilla", "median_rbc", "median_robust_grounded_plr"]}},
            ]},
            "vary_reg": {"$or": [
                        {"tags": {"$in": ["vary_reg"]}},
            ]},
            "vary_lr": {"$or": [
                        {"tags": {"$in": ["vary_lr"]}},
            ]},
            "active_vs_random": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_random"]}},
            ]},
            "active_vs_rbc": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_rbc"]}},
            ]},
            "active_vs_plr": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_plr"]}},
            ]},
            "active_vs_robustplr": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_robust_plr"]}},
            ]},
            "active_vs_grounded_robustplr": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_robust_grounded_plr"]}},
            ]},
            "active_vs_vanilla": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_vanilla"]}},
            ]},
            "active_vs_random_reset": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_random_reset"]}},
            ]},
            "active_vs_activeplr": {"$or": [
                        {"tags": {"$in": ["median_activerl", "median_activeplr"]}},
            ]},
}
GROUP_BY ={
            "debug": ["results_random", "results_activerl"],
            "results": ["results_random",  "results_activeplr", "results_activerl", "results_random_reset", "results_vanilla", "results_rbc", "results_robust_plr", "results_robust_grounded_plr"],
            "sim2real": ["median_random", "median_activeplr", "median_activerl", "median_robust_plr", "median_random_reset", "median_vanilla", "median_rbc", "median_robust_grounded_plr"],
            "vary_reg": ["reg_0", "reg_0.5", "reg_0.05", "reg_0.005"],
            "vary_lr": ["lr_1", "lr_0.1", "lr_0.01", "lr_0.001", "lr_0.0001"],
            "active_vs_random": ["median_activerl", "median_random"],
            "active_vs_rbc": ["median_activerl", "median_rbc"],
            "active_vs_plr": ["median_activerl", "median_plr"],
            "active_vs_vanilla": ["median_activerl", "median_vanilla"],
            "active_vs_random_reset": ["median_activerl", "median_random_reset"],
            "active_vs_activeplr": ["median_activerl", "median_activeplr"],
            "active_vs_robustplr": ["median_activerl", "median_robustplr"],
            "active_vs_grounded_robustplr": ["median_activerl", "median_robust_grounded_plr"]

}
BASELINE = {
            "results": "results_random",
            "sim2real": "median_random",
            "debug": "results_random",
}

COLORS = {
    "results": {
        "results_random": "red",
        "results_rbc": "blue",
        "results_activeplr": "green",
        "results_activerl": "orange",
        "results_robust_plr": "purple",
        "results_random_reset": "pink",
        "results_vanilla": "black",
        "results_robust_grounded_plr": "brown"
    },
    "sim2real": {
        "median_random": "red",
        "median_rbc": "blue",
        "median_activeplr": "green",
        "median_activerl": "orange",
        "median_robust_plr": "purple",
        "median_random_reset": "pink",
        "median_vanilla": "black",
        "median_robust_grounded_plr": "brown"
    },
    "vary_reg": {
        "reg_0": "red",
        "reg_0.5": "blue",
        "reg_0.05": "green",
        "reg_0.005": "orange",
    },
    "vary_lr": {
        "lr_1": "red",
        "lr_0.1": "blue",
        "lr_0.01": "green",
        "lr_0.001": "orange",
        "lr_0.0001": "purple",
    },
    "debug": {
        "results_random": "red",
        "results_activerl": "blue",
    },
    "active_vs_random": {
        "median_activerl": "blue",
        "median_random": "red",
    },
    "active_vs_rbc": {
        "median_activerl": "blue",
        "median_rbc": "red",
    },
    "active_vs_plr": {
        "median_activerl": "blue",
        "median_plr": "red",
    },
    "active_vs_robustplr": {
        "median_activerl": "blue",
        "median_plr": "red",
    },
    "active_vs_grounded_robustplr": {
        "median_activerl": "blue",
        "median_robust_grounded_plr": "brown",
    },
    "active_vs_vanilla": {
        "median_activerl": "blue",
        "median_vanilla": "red",
    },
    "active_vs_random_reset": {
        "median_activerl": "blue",
        "median_random_reset": "red",
    },
    "active_vs_activeplr": {
        "median_activerl": "blue",
        "median_activeplr": "red",
    },

}

NAMES = {
    "results": {
        "results_random": "Random",
        "results_rbc": "RBC",
        "results_activeplr": "Active-PLR",
        "results_activerl": "Active-RL",
        "results_robust_plr": "RPLR",
        "results_random_reset": "Domain Randomization",
        "results_vanilla": "RL",
        "results_robust_grounded_plr": "Grounded DR"
    },
    "sim2real": {
        "median_random": "Random",
        "median_rbc": "RBC",
        "median_activeplr": "Active-PLR",
        "median_activerl": "Active-RL",
        "median_robust_plr": "RPLR",
        "median_random_reset": "Domain Randomization",
        "median_vanilla": "RL",
        "median_robust_grounded_plr": "Grounded DR"
    },
    "vary_reg": {
        "reg_0": "γ=0",
        "reg_0.5": "0.5",
        "reg_0.05": "0.05",
        "reg_0.005": "0.005",
    },
    "vary_lr": {
        "lr_1": "1",
        "lr_0.1": "0.1",
        "lr_0.01": "0.01",
        "lr_0.001": "0.001",
        "lr_0.0001": "η=0.0001",
    },
    "debug": {
        "results_random": "Random",
        "results_activerl": "Active-RL",
    },
    "active_vs_random": {
        "median_activerl": "Active-RL",
        "median_random": "Random",
    },
    "active_vs_rbc": {
        "median_activerl": "Active-RL",
        "median_rbc": "RBC",
    },
    "active_vs_plr": {
        "median_activerl": "Active-RL",
        "median_plr": "PLR",
    },
    "active_vs_vanilla": {
        "median_activerl": "Active-RL",
        "median_vanilla": "RL",
    },
    "active_vs_robustplr": {
        "median_activerl": "Active-RL",
        "median_plr": "RPLR",
    },
    "active_vs_grounded_robustplr": {
        "median_activerl": "Active-RL",
        "median_robust_grounded_plr": "Grounded DR",
    },
    

}