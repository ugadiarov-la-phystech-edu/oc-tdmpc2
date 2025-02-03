import argparse
import json
import os

from omegaconf import OmegaConf
from tqdm import tqdm

from dlp import create_ddlp, load_checkpoint

os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings

warnings.filterwarnings('ignore')
from envs import make_env
import torch

from termcolor import colored

from common.seed import set_seed
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def collect(cfg: dict, checkpoint_path: str, save_folder: str, n_episodes: int):
    """
    Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task training)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `steps`: number of training/environment steps (default: 10M)
        `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
        $ python train.py task=mt80 model_size=48
        $ python train.py task=mt30 model_size=317
        $ python train.py task=dog-run steps=7000000
    ```
    """
    assert torch.cuda.is_available()
    cfg = OmegaConf.create(cfg)
    # assert cfg.steps > 0, 'Must train for at least 1 step.'
    torch.set_float32_matmul_precision('medium')

    # cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    model = None
    if cfg.obs == 'ddlp':
        ddlp_config_path = cfg.ddlp_config_path
        ddlp_checkpoint_path = cfg.ddlp_checkpoint_path
        model = create_ddlp(ddlp_config_path)
        model = load_checkpoint(model, ddlp_checkpoint_path)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        cfg.action_dim = model.action_dim

    env = make_env(cfg, extractor=model, save_folder=save_folder)
    agent = TDMPC2(cfg, ddlp_model=model)
    agent.load(checkpoint_path)

    for _ in tqdm(range(n_episodes)):
        obs, done, t = env.reset(), False, 0
        while not done:
            previous_actions = None
            if cfg.obs == 'ddlp' and cfg.transition_model_type != 'gnn':
                previous_actions = torch.from_numpy(env.get_actions()).to(obs['fg'].device)

            action = agent.act(obs, t0=t == 0, eval_mode=False, prev_actions=previous_actions)
            obs, reward, done, info = env.step(action)
            t += 1

    env.wait_for_futures()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as file_obj:
        config = json.load(file_obj)

    config = {k: v['value'] for k, v in config.items()}
    collect(config, args.checkpoint_path, args.save_folder, args.n_episodes)
