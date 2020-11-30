import tensorflow as tf
import glob
import os
import pickle
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt


# Get all event* runs from logging_dir subdirectories
logging_dirs = ['./storage/to_plot/GoToDoorAddressPPONumericPreNonlinearity', './storage/to_plot/GoToDoorAddressTextPPO']
plot_dir = './storage/plots/'
use_cache = True
event_paths = []
for logging_dir in logging_dirs:
    event_paths.extend(glob.glob(os.path.join(logging_dir, "event*")))

# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    try:
        for e in tqdm(tf.compat.v1.train.summary_iterator(path)):
            for v in e.summary.value:
                r = {'metric': v.tag, 'value': v.simple_value}
                runlog = runlog.append(r, ignore_index=True)
        metrics = runlog['metric'].unique()
        print(metrics)

    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None
    return runlog

def print_plot(metric, all_logs):
    ys = {}
    for method_name, logs in all_logs.items():
        data = logs[logs['metric'] == metric]["value"].tolist()
        plt.plot(data, label=method_name)

    plt.plot(ys.values())
    plt.ylabel(metric)
    plt.xlabel("Updates")
    plt.legend()
    plt.savefig(f"{plot_dir}/{metric}")
    plt.clf()

# Call & append
all_logs = {}
if not use_cache:
    for path in event_paths:
        method_name = path.split("/")[3]
        log = sum_log(path)
        all_logs[method_name] = log
    pickle.dump(all_logs, open("storage/all_logs.pkl", 'wb'))


if use_cache:
    all_logs = pickle.load(open("storage/all_logs.pkl", 'rb'))
values = print_plot("value", all_logs)
policy_loss = print_plot("policy_loss", all_logs)
value_loss = print_plot("value_loss", all_logs)
return_mean = print_plot("return_mean", all_logs)
rreturn_mean = print_plot("rreturn_mean", all_logs)
FPS = print_plot("FPS", all_logs)
