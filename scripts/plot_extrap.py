import glob
import os
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

# Get all event* runs from logging_dir subdirectories
logging_dir = '../storage/'
plot_dir = '../plots/extrap'
clrs = sns.color_palette("husl", 5)
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
use_cache = False
result_paths = []
eval_result_paths = []

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]



eval_results = {}
for root, exp_dirs, files in walklevel(logging_dir):
    for exp_dir in exp_dirs:
        for exp_dir_name in os.listdir(root + "/" + exp_dir):
            if exp_dir_name == "extrap.pkl":
                path = os.path.join(root, exp_dir, "extrap.pkl")
                eval_results[exp_dir] = pickle.load(open(path, "rb"))

# Call & append
fig, ax = plt.subplots()
results = {}
for method, data in eval_results.items():
    print(method)
    offsets = sorted([d['offset'] for d in data])
    return_means = []
    return_stds = []
    for setting in data:
        return_means.append(np.mean(setting['return_per_episode']))
        return_stds.append(np.std(setting['return_per_episode']))
    print(return_stds)
    print(return_means)

    ax.plot(offsets, return_means, label=method)  # c=clrs[i])
    ax.fill_between(offsets, return_means - np.array(return_stds)/2, return_means + np.array(return_stds)/2, alpha=0.3)  # , facecolor=clrs[i])

plt.ylabel("Return")
plt.xlabel("Updates")
plt.legend()
plt.savefig(f"{plot_dir}/extrapolate_returns.png")
plt.clf()

