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
parser.add_argument("--exp-id", required=True, type=str,
                    help="name of the experiment (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=123453,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--extrap-min", type=int, default=0,
                    help="minimum extrapolation offset")
parser.add_argument("--extrap-max", type=int, default=10,
                    help="maximum extrapolation offset")
parser.add_argument("--eval-one-model-per-seed", action="store_true", default=False,
                    help="sometimes we don't want to evaluate every seed for every model")

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

env = make_envs(args.procs, args.env, args.seed, args.extrap_min, args.extrap_min+1)

# Load agent
model_dirs = utils.get_models_for_exp(args.exp_id)
agents = defaultdict(list)
for model_dir in model_dirs:
    root = utils.get_model_dir(model_dir, args.exp_id)
    use_nac, use_text, use_memory = utils.get_args_for_model(model_dir)
    for idx, seed in enumerate(os.listdir(root)):
        exp_path = os.path.join(root, seed)
        if args.eval_one_model_per_seed and idx != 0:
            continue
        agents[exp_path].append(utils.Agent(env.observation_space, env.action_space, exp_path,
                            device=device, argmax=args.argmax, num_envs=args.procs,
                            use_memory=use_memory, use_text=use_text, use_nac=use_nac))
obs_space, preprocess_obss = utils.get_obss_preprocessor(env.envs[0].observation_space)
print("Agents loaded\n")


all_logs = defaultdict(list)
start_time = time.time()
for exp_path, agent_list in agents.items():
    for agent in agent_list:
        for offset in range(args.extrap_min, args.extrap_max):
            logs = {"offset": offset, "num_frames_per_episode": [], "return_per_episode": []}
            env = make_envs(args.procs, args.env, args.seed, offset, offset+1)
            obss = env.reset()

            log_done_counter = 0
            log_episode_return = torch.zeros(args.procs, device=device)
            log_episode_num_frames = torch.zeros(args.procs, device=device)

            while log_done_counter < args.episodes:
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
            fix_str = "/".join(exp_path.split("/")[-3:])
            all_logs[fix_str].append(logs)

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

# dump results
outfname = '/'.join(model_dirs[0].split('/')[:-1]) + "/evaluation.pkl"
pickle.dump(all_logs, open(f"{outfname}", "wb"))

# Print worst episodes
if args.worst_episodes_to_show > 0:
    print("\n{} worst episodes:".format(args.worst_episodes_to_show))
    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:args.worst_episodes_to_show]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
