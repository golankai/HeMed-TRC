import subprocess

# Define the range of seeds
seeds = range(42, 47)  # 42 to 46
# eval_train_sizes = [1500, 1750]
# eval_train_sizes = [25, 50, 75, 100, 250, 500, 750, 1000, 1078]
# eval_train_sizes = [100, 250, 500, 750, 1000, 1250, 1500]
eval_train_sizes = [25, 50, 75, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 1829]


# Loop over the seeds and run the script for each seed
# for seed in seeds:
#     print(f"Running run_exp.py with seed={seed}")
#     subprocess.run(["python", "run_exp.py", "--seed", str(seed)], check=True)

# Loop over eval train sizes
for eval_train_size in eval_train_sizes:
    print(f"Running run_exp.py with eval_train_size={eval_train_size}")
    subprocess.run(["python", "run_exp.py", "--max_eval_train_samples", str(eval_train_size)], check=True)