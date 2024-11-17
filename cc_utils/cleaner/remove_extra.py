import os
import shutil
from tqdm import tqdm

checkpoint_interval = 1000

base_path = "/home/chanb/scratch/simple_icl"
variant_name = "binary_synthetic-transformer-num_low_freq"

result_dir = os.path.join(base_path, "results", variant_name)
for root, _, filenames in os.walk(result_dir):
    if root.split("/")[-1] != "models":
        continue

    for filename in filenames:
        curr_checkpoint = os.path.join(root, filename)

        checkpoint = int(filename.split(".dill")[0])
        if checkpoint % checkpoint_interval != 0:
            os.remove(curr_checkpoint)
