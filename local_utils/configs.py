EXPERIMENTS = {
    "omniglot-input_noise": {
        "num_seeds": 3,
        "variants": [
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "input_noise_std",
                "values": [0.0, 0.1, 0.2]
            },
        ]
    },
}

EVALUATION = {
    "num_eval_samples": 3000,
    "batch_size": 100,
}
