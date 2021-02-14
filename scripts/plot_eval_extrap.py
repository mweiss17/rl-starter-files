import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D

fnames = []

fnames.append(os.path.join("storage", "extrap-10m", "GoToDoorAddressNumericNACPostNonlinearity", "seed_0", "eval_extrap_results.pkl"))
fnames.append(os.path.join("storage", "extrap-10m", "GoToDoorAddressNumeric", "seed_0", "eval_extrap_results.pkl"))


fig, ax = plt.subplots()

trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
trans = [trans1, trans2]

for idx, fname in enumerate(fnames):
    results = pickle.load(open(fname, "rb"))

    means = []
    stds = []

    for dic in results:
        means.append(dic['mean'])
        stds.append(dic['std'])
    x_vals = [x for x in range(0, len(means))]
    plt.errorbar(x_vals, means, stds, label=fname.split("/")[2], transform=trans[idx])
plt.legend()
plt.xlabel("House Number Offset")
plt.ylabel("Episode Reward")
plt.show()
# plt.savefig("/home/martin/Desktop/eval_extrap.png")
