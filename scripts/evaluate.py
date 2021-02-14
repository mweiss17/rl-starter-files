import os
import argparse
import time
import torch
import pickle
from torch_ac.utils.penv import ParallelEnv
from collections import defaultdict

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED) (storage/<model_name>/)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--use-number", action="store_true", default=False,
                    help="handle numerical input")
parser.add_argument("--use-nac", action="store_true", default=False,
                    help="use a neural accumulator")
parser.add_argument("--mem", action="store_true", default=False,
                    help="use a neural accumulator")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

def make_envs(procs, env_name, seed, extrap_min, extrap_max):
    envs = []
    for i in range(procs):
        env = utils.make_env(env_name, seed + 100000 * i, {"extrapolate_min": extrap_min, "extrapolate_max": extrap_max})
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")
    return env

extrap_min = 6
extrap_max = 100
env = make_envs(args.procs, args.env, args.seed, extrap_min, extrap_min+1)

# Load agent
root = utils.get_model_dir(args.model)
model_dirs = []
for d in os.listdir(root):
    model_dirs.append(os.path.join(root, d))
agent = utils.Agent(env.observation_space, env.action_space, model_dirs[0],
                    device=device, argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text, use_number=args.use_number, use_nac=args.use_nac)

print("Agent loaded\n")

# Initialize logs

all_logs = []

# Run agent

start_time = time.time()


obs_space, preprocess_obss = utils.get_obss_preprocessor(env.envs[0].observation_space, args.use_number)

for offset in range(extrap_min, extrap_max):
    logs = {"offset": offset, "num_frames_per_episode": [], "return_per_episode": []}
    env = make_envs(args.procs, args.env, args.seed, offset, offset+1)
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        # preprocessed_obss = preprocess_obss(obss, device=device)
        # agent.acmodel.forward(preprocessed_obss, [])
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)
        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask
    all_logs.append(logs)

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | dones: {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values(), log_done_counter))

# Print worst episodes
import pickle
outfname = '/'.join(model_dirs[0].split('/')[:-1]) + "/extrap.pkl"
pickle.dump(all_logs, open(f"{outfname}", "wb"))
n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
