import gym
import gym_minigrid


def make_env(env_key, seed=None, kwargs=None):
    if kwargs:
        env = gym.make(env_key, **kwargs)
    else:
        env = gym.make(env_key)
    env.seed(seed)
    return env
