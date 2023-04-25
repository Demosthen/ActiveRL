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