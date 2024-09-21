# TODO: Omniglot
# TODO: Simple ICL model
# TODO: More thorough experiments

# Section 5: Run for longer with added experiments
EXPERIMENTS = {
    # Section 5.1.1
    "synthetic-transformer-noisy_inputs": {
        "run_time": "01:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "input_noise_std",
                "values": [0.0, 0.02, 0.2, 0.4]
            }
        ]
    },
    "synthetic-transformer-p_high": {
        "run_time": "01:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.9, 0.99]
            },
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14, 2**16, 2**20]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-p_relevant": {
        "run_time": "01:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14, 2**16, 2**20]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
            },
        ]
    },
    # Section 5.1.3
    "synthetic-transformer-num_relevant_contexts": {
        "run_time": "01:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14, 2**16, 2**20]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "num_relevant_contexts",
                "values": [1, 2, 3, 4]
            }
        ]
    },
    # Section 5.1.4
    "synthetic-transformer-num_low_freq": {
        "run_time": "10:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "num_low_prob_classes",
                "values": [5, 45, 95, 495]
            }
        ]
    },
}

# Section 5
"""
EXPERIMENTS = {
    # Section 5.1.1
    "synthetic-transformer-noisy_inputs": {
        "run_time": "00:30:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "input_noise_std",
                "values": [0.0, 0.02, 0.2, 0.4]
            }
        ]
    },
    "synthetic-transformer-p_high": {
        "run_time": "00:30:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "p_high",
                "values": [0.5, 0.9, 0.99]
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
    "synthetic-transformer-p_relevant": {
        "run_time": "00:30:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
            },
        ]
    },
    # Section 5.1.2
    "synthetic-transformer-noisy_labels": {
        "run_time": "00:30:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "label_noise",
                "values": [0.001, 0.01, 0.1]
            }
        ]
    },
    # Section 5.1.3
    "synthetic-transformer-num_contexts": {
        "run_time": "01:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "num_contexts",
                "values": [1, 2, 4, 8, 16]
            }
        ]
    },
    "synthetic-transformer-num_relevant_contexts": {
        "run_time": "00:30:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**6, 2**8, 2**10, 2**12, 2**14]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "num_relevant_contexts",
                "values": [1, 2, 3, 4]
            }
        ]
    },
    # Section 5.1.4
    "synthetic-transformer-num_low_freq": {
        "run_time": "05:00:00",
        "num_seeds": 5,
        "variants": [
            {
                "key": "dataset_size",
                "values": [2**10, 2**12, 2**14, 2**16]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
            {
                "key": "num_low_prob_classes",
                "values": [5, 45, 95, 495]
            }
        ]
    },
}
"""

"""
# Preliminarty experiments on synthetic dataset
EXPERIMENTS = {
    "synthetic-transformer-context_len": {
        "run_time": "00:45:00",
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
                "key": "num_contexts",
                "values": [2, 4, 8]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-num_relevant_contexts": {
        "run_time": "00:45:00",
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
                "key": "num_relevant_contexts",
                "values": [1, 2, 3, 4]
            },
            {
                "key": "p_relevant_context",
                "values": [0.0, 0.9, 1.0]
            },
        ]
    },
    "synthetic-transformer-no_noise": {
        "run_time": "00:30:00",
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
        "run_time": "00:30:00",
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
        "run_time": "00:30:00",
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
        "run_time": "00:30:00",
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
        "run_time": "00:55:00",
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
    "synthetic-transformer-large_num_low_freq-input_noise_0.2": {
        "run_time": "00:55:00",
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
"""

EVALUATION = {
    "num_eval_samples": 3000,
    "batch_size": 100,
}
