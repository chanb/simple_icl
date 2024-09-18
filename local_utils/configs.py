EXPERIMENTS = {
    # "synthetic-transformer-noisy_labels": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.5, 0.75, 0.9, 0.99]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.0, 0.1, 0.5, 0.75, 0.9, 0.99, 1.0]
    #         },
    #     ]
    # },
    # "synthetic-transformer-noisy_inputs": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.9]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.0, 0.9, 1.0]
    #         },
    #     ]
    # },
    # "synthetic-transformer-no_noise": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.9]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.0, 0.9, 1.0]
    #         },
    #     ]
    # },
    "synthetic-transformer-context_len": {
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
    # "synthetic-iw_predictor": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.5, 0.75, 0.9, 0.99]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.0]
    #         },
    #     ]
    # },
    # "synthetic-ic_predictor": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.5, 0.75, 0.9, 0.99]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.1, 0.5, 0.75, 0.9, 0.99]
    #         },
    #     ]
    # },
    # "synthetic-alpha": {
    #     "num_seeds": 5,
    #     "variants": [
    #         {
    #             "key": "p_high",
    #             "values": [0.5, 0.75, 0.9, 0.99]
    #         },
    #         {
    #             "key": "dataset_size",
    #             "values": [2**6, 2**8, 2**10, 2**12, 2**14]
    #         },
    #         {
    #             "key": "p_relevant_context",
    #             "values": [0.1, 0.5, 0.75, 0.9, 0.99]
    #         },
    #     ]
    # },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
