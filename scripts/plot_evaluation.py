import os
import pickle
import argparse
import datetime
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D

parser = argparse.ArgumentParser()
parser.add_argument("--exp-id", type=str, help="name of the experiment (REQUIRED)")
args = parser.parse_args()

exp_id = args.exp_id if args.exp_id != None else str(datetime.datetime.today()).split(" ")[0]

root = os.path.join(os.getcwd(), "storage", exp_id)
results = pickle.load(open(os.path.join(root, "evaluation.pkl"), "rb"))

computed_results = defaultdict(dict)
all_lens = defaultdict(list)
for path, res in results.items():
    exp_id, model_name, seed = path.split("/")
    extracted = []
    for data in res:
        extracted.append(data['return_per_episode'])
        all_lens[model_name].append(len(data['return_per_episode']))
    computed_results[model_name][seed] = extracted

all_results = {}
for model_name, res in computed_results.items():
    min_len = min(all_lens[model_name])
    same_size_results = []
    for offset_runs in res.values():
        for run in offset_runs:
            same_size_results.append(run[:min_len])
    all_results[model_name] = np.array(same_size_results)
import pdb; pdb.set_trace()

fig, ax = plt.subplots()

trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
trans = [trans1, trans2]

for model_name, res in all_results.items():
    offsets_x_vals = [x for x in range(0, len(res[0]))]
    # plt.errorbar(x_vals, res.mean(axis=0), res.std(axis=0), label=model_name, transform=trans[idx])
    ax.plot(offsets_x_vals, res.mean(axis=0), label=model_name)  # c=clrs[i])
    ax.fill_between(offsets_x_vals, res.mean(axis=0) - res.std(axis=0)/2, res.mean(axis=0) + res.std(axis=0)/2, alpha=0.3)  # , facecolor=clrs[i])

plt.legend()
plt.xlabel("House Number Offset")
plt.ylabel("Episode Reward")
plt.savefig("eval_extrap.png")
