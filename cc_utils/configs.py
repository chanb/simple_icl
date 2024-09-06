EXPERIMENTS = {
    "simple_icl-transformer": {
        "run_time": "00:55:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            }
        ]
    },
    "simple_icl-transformer-non_linear": {
        "run_time": "00:55:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "num_high_prob_classes",
                "values": [100],
            },
            {
                "key": "num_low_prob_classes",
                "values": [100],
            }
        ]
    },
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
    "simple_icl-fixed_g-query_only_alpha": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": "high_prob",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "ground_truth_prob",
                "values": [ii / 10 for ii in range(0, 11)]
            },
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
                "values": [ii / 10 for ii in range(0, 11)]
            },
        ]
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
