# EXPERIMENTS = {
#     "synthetic-transformer-p_high": {},
#     "synthetic-transformer-noisy_inputs": {},
#     "synthetic-transformer-noisy_labels": {},
#     "synthetic-transformer-num_contexts": {},
#     "synthetic-transformer-num_low_freq": {},
#     "synthetic-transformer-num_relevant_contexts": {},
#     "synthetic-transformer-p_relevant": {},
# }

EXPERIMENTS = {
    "standard_basis": {
        "num_seeds": 3,
        "variants": [
            {
                "key": "num_contexts",
                "values": [2, 4]
            },
        ]
    },
}

# EXPERIMENTS = {
#     "omniglot-input_noise-context_len_4": {
#         "num_seeds": 1,
#         "variants": [
#             {
#                 "key": "dataset_size",
#                 "values": [100000, 1000000]
#             },
#             {
#                 "key": "p_relevant_context",
#                 # "values": [0.0, 0.9, 1.0]
#                 "values": [0.9]
#             },
#             {
#                 "key": "input_noise_std",
#                 "values": [0.1, 1.0]
#             },
#         ]
#     },
# }

# Preliminary
# EXPERIMENTS = {
#     "omniglot-input_noise": {
#         "num_seeds": 1,
#         "variants": [
#             {
#                 "key": "dataset_size",
#                 "values": [10000, 100000, 1000000]
#             },
#             {
#                 "key": "p_relevant_context",
#                 "values": [0.0, 0.9, 1.0]
#             },
#             {
#                 "key": "input_noise_std",
#                 "values": [0.1, 0.2]
#             },
#         ]
#     },
# }

EVALUATION = {
    "num_eval_samples": 3000,
    "batch_size": 100,
}
