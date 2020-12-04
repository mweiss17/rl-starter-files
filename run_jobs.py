import os
import datetime
import subprocess
import hydra
import yaml
import itertools
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict


SAMPLE_KEYS = {"list", "uniform", "range", "cartesian", "sequential", "chain"}
HYDRA_CONF_PATH = str(Path(__file__).parent / "configs/core.yaml")
np.random.seed(seed=0)

class RandomSearchError(Exception):
    pass

def now_str():
    """
    20200608_125339_353416
    """
    now = str(datetime.datetime.now())
    now = now.replace("-", "").replace(":", "").replace(" ", "_").replace(".", "_")
    return now

def printlines():
    print("=" * 80)
    print("=" * 80)

def get_hydra_args(opts, exclude=set()):
    hydra_args = ""
    for k, v in opts.items():
        if k not in exclude:
            if isinstance(v, list):
                v = f'"{v}"'
            hydra_args += f" {k}={v}"
    return hydra_args

def fill_mila_template(template_str, conf):
    """
    Formats the template_str with variables from the conf dict,
    which is a sampled experiment
    Args:
        template_str (str): sbatch template
        conf (dict): sbatch parameters
    Returns:
        str: formated template
    """
    user = os.environ.get("USER")
    home = os.environ.get("HOME")
    email = conf.get('email_id', "")

    partition = conf.get("partition", "main")
    cpu = conf.get("cpus", 6)
    # cpu constraints in long partition
    if partition == "long":
        cpu = min(cpu, 4)

    mem = conf.get("mem", 16)
    gres = conf.get("gres", "")
    time = str(conf.get("time", "4:00:00"))
    slurm_log = conf.get(
        "slurm_log", conf.get("base_dir", f"/network/tmp1/{user}/navi-slurm-%j.out")
    )
    env_name = conf.get("env_name", "navi-generalization")
    weights = conf.get("weights")
    code_loc = conf.get("code_loc", str(Path(home) / "navi-generalization/rl-starter-files/"))

    use_transformer = conf.get("use_transformer", True)
    workers = cpu - 1
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "navi-slurm-%j.out")
        if not Path(slurm_log).parent.exists() and not conf.get("dev"):
            Path(slurm_log).parent.mkdir(parents=True)

    if "dev" in conf and conf["dev"]:
        print(
            "Using:\n"
            + "\n".join(
                [
                    "  {:10}: {}".format("partition", partition),
                    "  {:10}: {}".format("cpus-per-task", cpu),
                    "  {:10}: {}".format("mem", mem),
                    "  {:10}: {}".format("gres", gres),
                    "  {:10}: {}".format("time", time),
                    "  {:10}: {}".format("slurm_log", slurm_log),
                    "  {:10}: {}".format("env_name", env_name),
                    "  {:10}: {}".format("code_loc", code_loc),
                ]
            )
        )

    partition = (
        f"#SBATCH --partition={partition}"
        if partition != "covid"
        else "#SBATCH --reservation=covid\n#SBATCH --partition=long"
    )
    cpu = f"#SBATCH --cpus-per-task={cpu}"
    mem = f"#SBATCH --mem={mem}GB"
    gres = f"#SBATCH --gres={gres}" if gres else ""
    time = f"#SBATCH --time={time}"
    email = f"#SBATCH --mail-user={email}"
    slurm_log = f"#SBATCH -o {slurm_log}\n#SBATCH -e {slurm_log}"
    return template_str.format(
        partition=partition,
        cpu=cpu,
        mem=mem,
        gres=gres,
        time=time,
        slurm_log=slurm_log,
        env_name=env_name,
        code_loc=code_loc,
        workers=workers,
        email=email
    )

def sample_param(sample_dict):
    """sample a value (hyperparameter) from the instruction in the
    sample dict:
    {
        "sample": "range | list",
        "from": [min, max, step] | [v0, v1, v2 etc.]
    }
    if range, as np.arange is used, "from" MUST be a list, but may contain
    only 1 (=min) or 2 (min and max) values, not necessarily 3
    Args:
        sample_dict (dict): instructions to sample a value
    Returns:
        scalar: sampled value
    """
    if not isinstance(sample_dict, dict) or "sample" not in sample_dict:
        return sample_dict

def sample_sequentials(sequential_keys, exp, idx):
    """
    Samples sequentially from the "from" values specified in each key
    of the experimental configuration which have sample == "sequential"
    Unlike `cartesian` sampling, `sequential` sampling iterates *independently*
    over each keys
    Args:
        sequential_keys (list): keys to be sampled sequentially
        exp (dict): experimental config
        idx (int): index of the current sample
    Returns:
        conf: sampled dict
    """
    conf = {}
    for k in sequential_keys:
        v = exp[k]["from"]
        conf[k] = v[idx % len(v)]
    return conf



def sample_cartesians(cartesian_keys, exp, idx):
    """
    Returns the `idx`th item in the cartesian product of all cartesian keys to
    be sampled.
    Args:
        cartesian_keys (list): keys in the experimental configuration that
            are to be used in the full cartesian product
        exp (dict): experimental configuration
        idx (int): index of the current sample
    Returns:
        dict: sampled point in the cartesian space (with keys = cartesian_keys)
    """
    conf = {}
    cartesian_values = [exp[key]["from"] for key in cartesian_keys]
    product = list(itertools.product(*cartesian_values))
    for k, v in zip(cartesian_keys, product[idx % len(product)]):
        conf[k] = v
    return conf


def sample_chains(chain_keys, exp, idx):
    """
    Returns the `idx`th item in the chain of all chain keys to be sampled.
    Args:
        chain_keys (list): keys in the experimental configuration
            that are to be used in the full chain
        exp (dict): experimental configuration
        idx (int): index of the current sample
    Returns:
        dict: sampled point in the cartesian space (with keys = chain_keys)
    """
    conf = {}
    chain_values = [[(key, value) for value in exp[key]["from"]] for key in chain_keys]
    chain = list(itertools.chain(*chain_values))
    k, v = chain[idx % len(chain)]
    conf[k] = v
    if exp[k].get("normalized"):
        conf["normalization_folder"] = k
    return conf


def get_uuid():
    return "{}_{}".format(np.random.randint(1e5, 1e6), np.random.randint(1e5, 1e6))


def sample_search_conf(exp, idx=0):
    """
    Samples parameters parametrized in `exp`: should be a dict with
    values which fit `sample_params(dic)`'s API
    Args:
        exp (dict): experiment's parametrization
        idx (int): experiment's idx in the sampling procedure (useful in case a key
            should be sampled in a cartesian or sequential manner)
    Returns:
        dict: sampled configuration
    """
    conf = {}
    cartesians = []
    sequentials = []
    chains = []
    for k, v in exp.items():

        if isinstance(v, dict) and v.get('sample') == 'cartesian':
            cartesians.append(k)
        elif isinstance(v, dict) and v.get('sample') == 'sequential':
            sequentials.append(k)
        elif isinstance(v, dict) and v.get('sample') == 'chain':
            chains.append(k)

    if sequentials:
        conf.update(sample_sequentials(sequentials, exp, idx))
    if chains:
        conf.update(sample_chains(chains, exp, idx))
    if cartesians:
        conf.update(sample_cartesians(cartesians, exp, idx))
    return conf


def compute_n_search(conf):
    """
    Compute the number of searchs to do if using -1 as n_search and using
    cartesian search
    Args:
        conf (dict): exprimental configuraiton
    Raises:
        RandomSearchError: Cannot be called if no cartesian or sequential field
    Returns:
        int: size of the cartesian product or length of longest sequential field
    """
    samples = defaultdict(list)
    for k, v in conf.items():
        if not isinstance(v, dict) or "sample" not in v:
            continue
        samples[v["sample"]].append(v)

    if "cartesian" in samples:
        total = 1
        for s in samples["cartesian"]:
            total *= len(s["from"])
        return total
    if "sequential" in samples:
        total = max(map(len, [s["from"] for s in samples["sequential"]]))
        return total
    if "chain" in samples:
        total = sum(map(len, [s["from"] for s in samples["chain"]]))
        return total

    raise RandomSearchError(
        "Used n_search=-1 without any field being 'cartesian' or 'sequential'"
    )

class NpEncoder(json.JSONEncoder):
    """
    Class to convert `obj` into json encodable objects.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



@hydra.main(config_path=HYDRA_CONF_PATH, strict=False)
def main(conf: DictConfig) -> None:

    """
                HOW TO USE
    $ python experiment.py exp_file=experiment n_search=20
    add `dev=True` to just see the commands that would be run, without
    running them
    NOTE: ALL parameters used in run.py may be overwritten from this commandline.
    For instance you can change init_fraction_sick
    $ python experiment.py exp_file=experiment n_search=20 init_fraction_sick=0.1
    NOTE: you may also pass arguments overwriting the default `sbatch` job's
    parametrization like partition, gres, code_loc (=where is the simulator's code),
    env_name (= what conda env to load). For instance:
    $ python experiment.py partition=unkillable gres=gpu:1 env_name=covid-env\
                              n_search=20 init_fraction_sick=0.1
    """
    # These will be filtered out when passing arguments to run.py
    RANDOM_SEARCH_SPECIFIC_PARAMS = {
        "n_search",  # number of random iterations
        "n_runs_per_search",  # number of random iterations
        "dev",  # dev-mode: print stuff, don't run them
        "exp_file",  # what experimental parametrization
        "partition",  # sbatch partition to use
        "cpus",  # sbatch number of cpus
        "mem",  # sbatch memory to request
        "time",  # sbatch job upper bound on duration
        "slurm_log",  # sbatch logs destination
        "gres",  # sbatch gres arg, may be nothing or gpu:1
        "env_name",  # conda environment to load
        "code_loc",  # where to find the source code, will cd there
        "weights",  # where to find the transformer's weights
        "now_str",  # naming scheme
        "parallel_search",  # run with & at the end instead of ; to run in subshells
        "start_index",  # ignore the first runs, to continue an exploration for instance
        "use_tmpdir",  # use SLURM_TMPDIR and copy files to outdir after
        "weights_dir",  # where are the weights
        "base_dir",  # output dir will be base_dir/tracing_method
        "normalization_folder",  # if this is a normalization run
        "exp_name",  # folder name in base_dir => base_dir/exp_name/method/...
        "email_id", # email id where you can receive notifications regarding jobs (began, completed, failed)
    }

    # move back to original directory because hydra moved
    os.chdir(hydra.utils.get_original_cwd())

    # get command-line arguments as native dict
    overrides = OmegaConf.to_container(conf, resolve=True)

    # load experimental configuration
    # override with exp_file=<X>
    # where <X> is in configs/exp and is ".yaml"
    exp_file_path = (
        Path(__file__).resolve().parent
        / "configs"
        / "exp"
        / (overrides.get("exp_file", "randomization") + ".yaml")
    )
    with exp_file_path.open("r") as f:
        conf = yaml.safe_load(f)
    # override experimental parametrization with the commandline conf
    conf.update(overrides)

    # -------------------------------------
    # -----  Compute Specific Values  -----
    # -------------------------------------

    conf["n_runs_per_search"] = conf.get("n_runs_per_search", 1)

    if conf.get("n_search") == -1:
        total_runs = compute_n_search(conf)
        conf["n_search"] = total_runs // conf["n_runs_per_search"]
    else:
        total_runs = conf["n_runs_per_search"] * conf["n_search"]

    if total_runs % conf["n_runs_per_search"] != 0:
        raise RandomSearchError(
            "n_search ({}) is not divisible by n_runs_per_epoch ({})".format(
                total_runs, conf["n_runs_per_search"]
            )
        )

    if "exp_name" in conf:
        if "base_dir" in conf:
            conf["base_dir"] = str(Path(conf["base_dir"]) / conf["exp_name"])
            print(f"Running experiments in base_dir: {conf['base_dir']}")
        else:
            print(f"Ignoring 'exp_name' {conf['exp_name']} as no base_dir was provided")

    print(f"Running {total_runs} scripts")

    conf["now_str"] = now_str()
    parallel_search = conf.get("parallel_search", False)
    start_index = conf.get("start_index", 0)
    base = Path(__file__).resolve().parent
    with (base / "configs/mila_sbatch_template.sh").open("r") as f:
        template_str = f.read()

    use_tmpdir = conf.get("use_tmpdir", False)
    dev = "dev" in conf and conf["dev"]
    is_tune = conf.get("tune", False)
    sampled_keys = [k for k, v in conf.items() if isinstance(v, dict) and "sample" in v]
    sampled_str = "\n".join([f"  {k}: {{{k}}}" for k in sampled_keys])

    if is_tune and use_tmpdir:
        raise RandomSearchError("Cannot use tune and $SLURM_TMPDIR together")
    if use_tmpdir and not conf["outdir"]:
        raise RandomSearchError(
            "Using $SLURM_TPMDIR but no `outdir` has been specified"
        )

    home = os.environ["HOME"]
    copy_dest = conf["outdir"] if "outdir" in conf else conf["base_dir"]
    if not dev:
        Path(copy_dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(exp_file_path, Path(copy_dest) / exp_file_path.name)

    # run n_search jobs
    printlines()
    old_opts = set()
    run_idx = start_index
    for i in range(conf.get("n_search", 1)):
        print("\nJOB", i)

        # fill template
        job_str = fill_mila_template(template_str, conf)

        # do n_runs_per_search simulations per job
        for k in range(conf.get("n_runs_per_search", 1)):
            skipped = False
            opts = sample_search_conf(conf, run_idx)
            run_idx += 1

            opts_str = json.dumps(opts, sort_keys=True, cls=NpEncoder)
            if opts_str in old_opts:
                print("\n Ran this job already ... skipping!")
                skipped = True
                continue

            old_opts.add(opts_str)

            # --------------------------------
            # -----  Use SLURM_TMPDIR ?  -----
            # --------------------------------
            if use_tmpdir:
                outdir = str(opts["outdir"])
                if not dev:
                    Path(outdir).resolve().mkdir(parents=True, exist_ok=True)
                opts["outdir"] = "$SLURM_TMPDIR"


            # convert params to string command-line args
            exclude = RANDOM_SEARCH_SPECIFIC_PARAMS

            command_suffix = "&\nsleep 5;\n" if parallel_search else ";\n"

            # append run command
            job_str += f"\n cd {conf['code_loc']}"
            job_str += f"\n python scripts/train.py --algo ppo "
            job_str += f"--env {conf['env']} "
            job_str += f"--model {conf['model']} "
            job_str += f"--seed {opts['seed']} "
            job_str += f"--procs {conf['procs']} "
            job_str += f"--save-interval 10 "
            job_str += f"--frames {conf['frames']} "
            job_str += f"{'--text' if conf['text'] else '' } "
            job_str += f"{'--use-number' if conf['number'] else '' } "
            job_str += f"{'--use-nac' if conf['use_nac'] else '' } "
            job_str += f"{'--eval-env' if conf['eval_env'] else '' } "

            job_str += command_suffix

        if skipped:
            continue

        # create temporary sbatch file
        tmp = Path(tempfile.NamedTemporaryFile(suffix=".sh").name)
        # give somewhat meaningful name to t
        tmp = tmp.parent / (Path(opts.get("outdir", "")).name + "_" + tmp.name)
        if not dev:
            with tmp.open("w") as f:
                f.write(job_str)

        command = f"sbatch {str(tmp)}"

        # dev-mode: don't actually run the command
        if dev:
            print("\n>>> ", command, end="\n\n")
            print(str(tmp))
            print("." * 50)
            print(job_str)
        else:
            # not dev-mode: run it!
            _ = subprocess.call(command.split(), cwd=home)
            # print("In", opts["outdir"])
            print("With Sampled Params:")
            print(sampled_str.format(**{k: opts.get(k) for k in sampled_keys}))

        # prints
        print()
        printlines()


if __name__ == "__main__":
    main()
