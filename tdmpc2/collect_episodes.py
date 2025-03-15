import os
import random

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings

warnings.filterwarnings('ignore')
import torch

import hydra

from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def schedule(current_episode, start_episode, end_episode, start_value, end_value, ):
    if current_episode <= start_episode:
        return start_value
    elif current_episode >= end_episode:
        return end_value
    else:
        return start_value + (current_episode - start_episode) / (end_episode - start_episode) * (
                    end_value - start_value)


@hydra.main(config_name='config', config_path='.')
def collect(cfg: dict):
    torch.set_float32_matmul_precision('medium')
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    if cfg.get('checkpoint', None):
        print(f'Loading checkpoint: {cfg.checkpoint}')
        state_dict = torch.load(cfg.checkpoint)
        agent.load(state_dict)

    run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                     config=OmegaConf.to_container(cfg, resolve=True))
    total_episodes = cfg.n_train_episodes + cfg.n_val_episodes
    for episode_id in tqdm(range(total_episodes), position=tqdm._get_free_pos(), desc='# Run episodes'):
        epsilon = schedule(episode_id, start_episode=0, end_episode=total_episodes - 1,
                           start_value=cfg.epsilon_greedy_start, end_value=cfg.epsilon_greedy_end)
        noise_scale = schedule(episode_id, start_episode=0, end_episode=total_episodes - 1,
                               start_value=cfg.noise_scale_start, end_value=cfg.noise_scale_end)
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        t = 0
        while not done:
            action = agent.act(obs, t0=t == 0, eval_mode=True)
            if random.random() < epsilon:
                action = env.action_space.sample()
                action = torch.as_tensor(action)
            else:
                action += noise_scale * torch.randn_like(action)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1

        record = {'return': ep_reward}
        if 'success' in info:
            record['success'] = info['success']

        run.log(record)

    env.close()
    run.finish()


if __name__ == '__main__':
    collect()
