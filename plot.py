import tensorflow as tf
import glob
import os
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Get all event* runs from logging_dir subdirectories
logging_dir = './storage/'
plot_dir = './plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
use_cache = False
event_paths = []

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

for root, exp_dirs, files in walklevel(logging_dir):
    for exp_dir in exp_dirs:
        for seed_dir in  os.listdir(root + "/" + exp_dir):
            event_paths.extend(glob.glob(os.path.join(root, exp_dir, seed_dir, "results.pkl")))


def print_plot(metric, all_logs):

    for method_name, logs in all_logs.items():
        data = []
        for seed in range(len(logs)):
            data.append(logs[seed][metric])
        import pdb; pdb.set_trace()
        data = np.array(data)
        plt.errorbar(range(len(data[0])), data.mean(axis=0), yerr=data.std(axis=0), label=method_name)

    plt.ylabel(metric)
    plt.xlabel("Updates")
    plt.legend()
    plt.savefig(f"{plot_dir}/{metric}")
    plt.clf()


# Call & append
all_logs = {}
if not use_cache:
    for path in event_paths:
        method_name = path.split("/")[2]
        seed = path.split("/")[3]
        if not all_logs.get(method_name):
            all_logs[method_name] = []
        log = pickle.load(open(path, 'rb'))
        all_logs[method_name].append(log)
    pickle.dump(all_logs, open("storage/all_logs.pkl", 'wb'))

if use_cache:
    all_logs = pickle.load(open("storage/all_logs.pkl", 'rb'))

values = print_plot("value", all_logs)
policy_loss = print_plot("policy_loss", all_logs)
value_loss = print_plot("value_loss", all_logs)
return_mean = print_plot("return_mean", all_logs)
rreturn_mean = print_plot("rreturn_mean", all_logs)
FPS = print_plot("FPS", all_logs)
