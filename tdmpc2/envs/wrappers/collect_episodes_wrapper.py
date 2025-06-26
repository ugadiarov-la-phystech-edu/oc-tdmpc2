import collections
import concurrent.futures
import os
import random

import gym
import numpy as np
from PIL import Image
from tqdm import tqdm


def _save_episode(dataset_path, split, split_info, episode_id, episode_observations, episode_actions, episode_rewards):
    split_path = os.path.join(dataset_path, split)
    os.makedirs(split_path, exist_ok=True)
    np.save(os.path.join(split_path, 'info.npy'), split_info, allow_pickle=True)

    episode_info = split_info[episode_id]
    actions_path = os.path.join(split_path, episode_info['actions'])
    episode_path = os.path.dirname(actions_path)
    os.makedirs(episode_path, exist_ok=True)
    np.save(actions_path, np.asarray(episode_actions), allow_pickle=True)

    rewards_path = os.path.join(split_path, episode_info['rewards'])
    np.save(rewards_path, np.asarray(episode_rewards), allow_pickle=True)

    for obs_relative_path, observation in zip(split_info[episode_id]['obs'], episode_observations):
        obs_path = os.path.join(split_path, obs_relative_path)
        Image.fromarray(observation).save(obs_path)


class CollectEpisodes(gym.Wrapper):
    def __init__(self, env, cfg,):
        super().__init__(env)
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path
        total_episodes = (cfg.n_val_episodes + cfg.n_train_episodes)
        self.val_fraction = cfg.n_val_episodes / total_episodes
        self._dataset_info = dict([(split, {'version': 2}) for split in ('train', 'val')])
        self.counter = collections.Counter()
        self.episode_id = -1
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.futures = collections.deque()
        self.max_futures = 10
        self.pbar = tqdm(total=total_episodes, position=tqdm._get_free_pos(), desc='# Saved episodes')

    def reset(self):
        if self.episode_id >= 0:
            self.save_episode()

        self.episode_id += 1

        obs = super().reset()
        self.episode_observations.append(np.copy(self.env.get_last_source_frame()))

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.episode_observations.append(np.copy(self.env.get_last_source_frame()))
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

        return obs, reward, done, info

    def save_episode(self):
        split = 'val' if random.random() <= self.val_fraction else 'train'
        split_info = self._dataset_info[split]
        episode_id = self.counter[split]
        self.counter[split] += 1

        episode_folder_path = f'obs/ep_{episode_id}'
        observation_paths = [f'{episode_folder_path}/s_{i}.JPEG' for i in range(len(self.episode_observations))]
        actions_path = f'{episode_folder_path}/actions.npy'
        rewards_path = f'{episode_folder_path}/rewards.npy'
        episode_info = {'obs': observation_paths, 'actions': actions_path, 'rewards': rewards_path}
        split_info['num_episodes'] = self.counter[split]
        split_info[episode_id] = episode_info

        while (len(self.futures) > 0 and self.futures[0].done()) or len(self.futures) > self.max_futures:
            self.futures.popleft().result()
            self.pbar.update(1)

        future = self.executor.submit(_save_episode, self.dataset_path, split, split_info, episode_id,
                                      self.episode_observations, self.episode_actions, self.episode_rewards)
        self.futures.append(future)
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []

    def close(self):
        if len(self.episode_observations) > 0:
            assert len(self.episode_actions) > 0
            assert len(self.episode_rewards) > 0
            self.save_episode()

        for future in self.futures:
            future.result()
            self.pbar.update(1)

        self.pbar.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
