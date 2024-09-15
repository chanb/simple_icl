EXPERIMENTS = {
    "synthetic-transformer": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
    "synthetic-iw_predictor": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
    "synthetic-ic_predictor": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
    "synthetic-alpha": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            },
        ]
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
