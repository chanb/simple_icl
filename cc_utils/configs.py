EXPERIMENTS = {
    "simple_icl-learned_g": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            }
        ]
    },
    "simple_icl-fixed_g": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "ground_truth_prob",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
    "simple_icl-fixed_g-context_alpha": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "ground_truth_prob",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
