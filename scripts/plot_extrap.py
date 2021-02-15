import glob
import os
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
import utils
from matplotlib import pyplot as plt

model_name = 'ACMLP'
logging_dir = os.path.join(os.getcwd(), 'storage', model_name)
plot_dir = os.path.join(os.getcwd(), 'plots', model_name, 'extrap')


if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

clrs = sns.color_palette("husl", 5)

use_cache = False
result_paths = []
eval_result_paths = []

# Get all event* runs from logging_dir subdirectories
eval_results = {}
for root, exp_dirs, files in utils.walklevel(logging_dir):
    # for exp_dir in exp_dirs:
    #     for exp_dir_name in os.listdir(root + "/" + exp_dir):
    #         import pdb; pdb.set_trace()
    #         if exp_dir_name == "evaluation.pkl":
    #             path = os.path.join(root, exp_dir, "evaluation.pkl")
    #             eval_results[exp_dir] = pickle.load(open(path, "rb"))
    for filename in files:
        if filename == "evaluation.pkl":
            path = os.path.join(root, "evaluation.pkl")
            eval_results["root"] = pickle.load(open(path, "rb"))

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

    ax.plot(offsets, return_means, label=method)  # c=clrs[i])
    ax.fill_between(offsets, return_means - np.array(return_stds)/2, return_means + np.array(return_stds)/2, alpha=0.3)  # , facecolor=clrs[i])

plt.ylabel("Return")
plt.xlabel("Updates")
plt.legend()
plt.savefig(f"{plot_dir}/extrapolate_returns.png")
plt.clf()

