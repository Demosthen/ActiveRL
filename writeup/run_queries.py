QUERIES = {
            "results": {"$or": [
                        {"tags": {"$in": ["results"]}},
            ]},
            "sim2real": {"$or": [
                        {"tags": {"$in": ["median_random", "median_activeplr", "median_activerl", "median_plr", "median_random_reset", "median_vanilla", "median_rbc"]}},
            ]},
}
GROUP_BY ={
            "results": ["results_random",  "results_activeplr", "results_activerl", "results_plr", "results_random_reset", "results_vanilla"],
            "sim2real": ["median_random", "median_activeplr", "median_activerl", "median_plr", "median_random_reset", "median_vanilla", "median_rbc"],
}
BASELINE = {
            "results": "results_random",
            "sim2real": "median_random",
}

COLORS = {
    "results": {
        "results_random": "red",
        "results_rbc": "blue",
        "results_activeplr": "green",
        "results_activerl": "orange",
        "results_plr": "purple",
        "results_random_reset": "pink",
        "results_vanilla": "black",
    },
    "sim2real": {
        "median_random": "red",
        "median_rbc": "blue",
        "median_activeplr": "green",
        "median_activerl": "orange",
        "median_plr": "purple",
        "median_random_reset": "pink",
        "median_vanilla": "black",
    }
}

NAMES = {
    "results": {
        "results_random": "Random",
        "results_rbc": "RBC",
        "results_activeplr": "Active-PLR",
        "results_activerl": "Active-RL",
        "results_plr": "PLR",
        "results_random_reset": "Domain Randomization",
        "results_vanilla": "RL",
    },
    "sim2real": {
        "median_random": "Random",
        "median_rbc": "RBC",
        "median_activeplr": "Active-PLR",
        "median_activerl": "Active-RL",
        "median_plr": "PLR",
        "median_random_reset": "Domain Randomization",
        "median_vanilla": "RL",
    }
}