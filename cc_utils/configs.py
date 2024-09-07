EXPERIMENTS = {
    # "simple_icl-transformer": {
    #     "run_time": "00:55:00",
    #     "num_seeds": 10,
    #     "variants": [
    #         {
    #             "key": "high_prob",
    #             "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
    #         }
    #     ]
    # },
    # "simple_icl-transformer-non_linear": {
    #     "run_time": "00:55:00",
    #     "num_seeds": 10,
    #     "variants": [
    #         {
    #             "key": "high_prob",
    #             "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
    #         },
    #         {
    #             "key": "num_high_prob_classes",
    #             "values": [100],
    #         },
    #         {
    #             "key": "num_low_prob_classes",
    #             "values": [100],
    #         }
    #     ]
    # },
    # "simple_icl-learned_g": {
    #     "run_time": "00:45:00",
    #     "num_seeds": 10,
    #     "variants": [
    #         {
    #             "key": "high_prob",
    #             "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
    #         }
    #     ]
    # },
    "simple_icl-fixed_g": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variants": [
            {
                "key": ["high_prob", "high_freq_prob", "low_freq_prob"],
                "values": [
                    [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
                    [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
                    [1 - 0.5, 1 - 0.67, 1 - 0.75, 1 - 0.8, 1 - 0.9, 1 - 0.99],
                ],
            }
        ],
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
