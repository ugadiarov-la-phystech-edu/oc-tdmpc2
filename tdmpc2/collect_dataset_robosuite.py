import argparse
import collections
import concurrent.futures
import multiprocessing
import os

import numpy as np
import tqdm
from PIL import Image
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from envs.robosuite_env import RobosuiteEnv


def make_env(seed):
    env_config = OmegaConf.load("config.yaml")
    env_config["seed"] = 0
    env_config["obs_size"]
    env = RobosuiteEnv()
    env.action_space.seed(seed)
    env = TimeLimit(env, env.unwrapped._max_episode_length)
    return env


def collect(env_path, seed):
    env = make_env(env_path, seed)
    observations = [env.reset()]
    actions = []
    rewards = []
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(rew)

    env.close()
    return np.stack(observations), np.stack(actions), np.stack(rewards)


def store(observations, actions, rewards, output_dir, folder, episode_id):
    episode_path = os.path.join(output_dir, folder, f'{episode_id:05d}')
    os.makedirs(episode_path)
    np.save(os.path.join(episode_path, 'actions.npy'), actions)
    np.save(os.path.join(episode_path, 'rewards.npy'), rewards)
    for i, observation in enumerate(observations):
        Image.fromarray(observation).save(os.path.join(episode_path, f'{i:04d}.png'))

    return observations.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--env_config_path', type=str, required=True)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    args = parser.parse_args()

    context = multiprocessing.get_context('forkserver')
    collect_futures = collections.deque()
    store_futures = collections.deque()
    collected = 0
    seed = args.seed
    total = args.train_size + args.val_size
    val_portion = args.val_size / total
    counter = collections.Counter({'train': 0, 'val': 0})
    collect_tqdm_bar = tqdm.tqdm(total=total, smoothing=0, desc='Collected observations ')
    store_tqdm_bar = tqdm.tqdm(total=total, smoothing=0, desc='Stored observations ')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers, mp_context=context) as executor:
        while collected < total:
            future = executor.submit(collect, args.env_config_path, seed)
            collect_futures.append(future)
            seed += 1

            if len(collect_futures) >= args.n_workers or collect_futures[0].done():
                observations, actions, rewards = collect_futures.popleft().result()
                folder = 'val' if np.random.sample() <= val_portion else 'train'
                collected += observations.shape[0]
                collect_tqdm_bar.update(observations.shape[0])
                future = executor.submit(store, observations, actions, rewards, args.output_dir, folder, counter[folder])
                store_futures.append(future)
                counter[folder] += 1

            while len(store_futures) > 0 and store_futures[0].done():
                stored = store_futures.popleft().result()
                store_tqdm_bar.update(stored)

        for future in collect_futures:
            future.cancel()

        while len(store_futures) > 0:
            stored = store_futures.popleft().result()
            store_tqdm_bar.update(stored)

    collect_tqdm_bar.close()
    store_tqdm_bar.close()
