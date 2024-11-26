import os
import shutil
from tqdm import tqdm

match_name = "100000.dill"

base_path = "/home/chanb/scratch/simple_icl"
# variant_name = "binary_synthetic-transformer-num_low_freq"
variant_name = "synthetic-transformer-4_tf_blocks-noisy_inputs"

dat_path = os.path.join(base_path, "configs", "{}.dat".format(variant_name))
result_dir = os.path.join(base_path, "results", variant_name)
print(dat_path)
print(result_dir)

result_map = dict()
for result_path in os.listdir(result_dir):
    experiment_name = "-".join(result_path.split("-")[:-8])
    result_map[experiment_name] = result_path

print(len(result_map))

new_dat_content = ""
num_runs = 0
new_num_runs = 0
for dat_line in tqdm(open(dat_path)):
    experiment_name = dat_line.split("/")[-1].split(".json")[0]

    num_runs += 1
    if experiment_name not in result_map:
        new_dat_content += dat_line
        new_num_runs += 1
        continue

    experiment_path = os.path.join(result_dir, result_map[experiment_name])

    is_completed = False
    for _, _, filenames in os.walk(experiment_path):
        if match_name in filenames:
            is_completed = True
            break

    if not is_completed:
        shutil.rmtree(experiment_path)
        new_dat_content += dat_line
        new_num_runs += 1

print(new_num_runs, num_runs)

with open(dat_path, "w") as f:
    f.writelines(new_dat_content)