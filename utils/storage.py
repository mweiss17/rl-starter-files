import csv
import os
import torch
import logging
import sys

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name, exp_id):
    return os.path.join(get_storage_dir(), exp_id, model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def get_models_for_exp(exp_id):
    # Get all event* runs from logging_dir subdirectories
    root = os.path.join(os.getcwd(), "storage", exp_id)
    model_folder_paths = [os.path.join(root, p) for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    return model_folder_paths

def get_args_for_model(model_dir):
    # e.g. storage/<date>/<model_dir>
    seed_paths = [os.path.join(model_dir, p) for p in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, p))]
    logfile = os.path.join(seed_paths[0], "log.txt")
    use_nac = False
    use_text = False
    use_memory = False
    with open(logfile, 'r') as f:
        line = f.read()
        if "use-nac" in line:
            use_nac = True
        if "text" in line:
            use_text = True
        if "memory" in line:
            use_memory = True
    return use_nac, use_text, use_memory

