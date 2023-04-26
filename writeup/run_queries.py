QUERIES = {
            "results": {"$or": [
                        {"tags": {"$in": ["results"]}},
            ]},
}
GROUP_BY ={
            "results": ["results_random",  "results_activeplr", "results_activerl", "results_plr", "results_random_reset", "results_vanilla"],
}
BASELINE = {
            "results": "results_random",
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
    }
}