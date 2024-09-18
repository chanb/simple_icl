EXPERIMENTS = {
    "synthetic-transformer-context_len": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "context_len",
                "values": [2, 4, 8]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-no_noise": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-noisy_inputs_0.2": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-noisy_labels_0.1": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-noisy_labels_0.01": {
        "run_time": "00:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-large_num_low_freq": {
        "run_time": "01:15:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.9]
            },
            {
                "key": "dataset_size",
                "values": [2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
