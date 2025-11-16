import collections
import itertools
import math
import os
import random
from pathlib import Path

import comet_ml
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline.dataset import EpisodesSlotsDataset
from sold_tdmpc2 import SoldTDMPC2


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataloader(source_root, slots_root, mode, sample_length, slots_file_name, batch_size, num_workers, drop_last=None):
    dataset = EpisodesSlotsDataset(source_root, slots_root, mode, sample_length, slots_file_name)
    is_train = mode == "train"
    if drop_last is None:
        drop_last = is_train

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, drop_last=drop_last)
    return dataloader


def step(agent, batch, num_cut_frames, probs, do_update, slot_norm=1.):
    obs = batch.z.swapaxes(0, 1)
    action = torch.nn.functional.pad(batch.action, pad=(0, 0, 1, 0), value=torch.nan).swapaxes(0, 1)
    reward = torch.nn.functional.pad(batch.reward, pad=(1, 0), value=torch.nan).swapaxes(0, 1).unsqueeze(-1)
    n = np.random.choice(num_cut_frames, p=probs)
    obs = obs[n:]
    action = action[n:]
    reward = reward[n:]
    is_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(do_update)
    statistics = agent.step(obs, action, reward, None, do_update=do_update)
    statistics['consistency_loss_relative'] = statistics['consistency_loss'] / slot_norm
    torch.set_grad_enabled(is_grad_enabled)
    return statistics


def get_num_cut_frames(sold_dynamics_num_context, use_variable_sequence_length):
    if not use_variable_sequence_length:
        return np.asarray([0]), np.asarray([1.], dtype=np.float32)

    num_cut_frames = np.arange(sold_dynamics_num_context)
    probs = [0.05] * (sold_dynamics_num_context - 1)
    probs.append(1 - sum(probs))
    return num_cut_frames, np.asarray(probs, dtype=np.float32)[::-1]


@hydra.main(config_name='train_sold_tdmpc2_offline', config_path='..')
def main(cfg: dict):
    set_random_seed(cfg.seed)
    sequence_length = cfg.horizon + cfg.sold_dynamics_num_context
    num_cut_frames, probs = get_num_cut_frames(cfg.sold_dynamics_num_context, cfg.use_variable_sequence_length)
    val_dataloader = make_dataloader(cfg.source_root, cfg.slots_root, 'val', sequence_length, cfg.slots_file_name,
                                     cfg.batch_size, cfg.num_workers)
    train_dataloader = make_dataloader(cfg.source_root, cfg.slots_root, 'train', sequence_length, cfg.slots_file_name,
                                       cfg.batch_size, cfg.num_workers)

    cfg.action_dim = val_dataloader.dataset.action_dim[0]
    assert len(cfg.action_lower_bound) == len(cfg.action_upper_bound), f'{len(cfg.action_lower_bound)} != {len(cfg.action_upper_bound)}'
    assert len(cfg.action_lower_bound) == cfg.action_dim, f'{len(cfg.action_lower_bound)} != {cfg.action_dim}'
    cfg.multitask = False
    cfg.obs_shape = {'slots': val_dataloader.dataset.slot_dim}
    cfg.episode_length = cfg.time_limit
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.num_epochs = math.inf if cfg.num_epochs is None else cfg.num_epochs
    agent = SoldTDMPC2(cfg)

    run_id = cfg.get("wandb_run_id", None)
    experiment = comet_ml.start(
        project_name=cfg.wandb_project,
        experiment_key=run_id,
        mode="get" if run_id else "create",
    )
    experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))
    experiment.set_name(cfg.wandb_run_name)

    start_epoch = 0
    n_train_dataloader = len(train_dataloader)
    n_val_dataloader = len(val_dataloader)
    work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.wandb_project / cfg.wandb_run_name
    total_loss = math.inf
    slot_norm = train_dataloader.dataset.get_slot_norm()
    for epoch in itertools.count(start=start_epoch, step=1):
        if epoch >= cfg.num_epochs:
            break

        with tqdm(total=n_train_dataloader, desc=f"Epoch: {epoch}. Train:") as pbar_train:
            train_statistics = collections.Counter()
            for i, batch in enumerate(train_dataloader):
                batch = batch.to(cfg.device)
                statistics = step(agent, batch, num_cut_frames, probs, do_update=True, slot_norm=slot_norm)
                pbar_train.set_postfix(statistics)
                pbar_train.update()
                train_statistics.update(statistics)
                global_step = i + epoch * n_train_dataloader
                statistics['global_step'] = global_step
                if global_step % cfg.log_every_batches == 0:
                    experiment.log_metrics({f'train/{k}': v for k, v in statistics.items()}, step=global_step, epoch=epoch)

            train_statistics = {f'train/{k}_epoch': v / n_train_dataloader for k, v in train_statistics.items()}
            train_statistics['epoch'] = epoch
            train_statistics['global_step'] = global_step
            if epoch % cfg.log_every_epochs == 0:
                experiment.log_metrics(train_statistics, step=global_step, epoch=epoch)

        with tqdm(total=n_val_dataloader, desc=f"Epoch: {epoch}. Val:") as pbar_val:
            val_statistics = collections.Counter()
            for i, batch in enumerate(val_dataloader):
                batch = batch.to(cfg.device)
                statistics = step(agent, batch, num_cut_frames, probs, do_update=False)
                pbar_val.set_postfix(statistics)
                pbar_val.update()
                val_statistics.update(statistics)

            val_statistics = {f'val/{k}': v / n_val_dataloader for k, v in val_statistics.items()}
            val_statistics['epoch'] = epoch
            val_statistics['global_step'] = global_step
            experiment.log_metrics(val_statistics, step=global_step, epoch=epoch)

        if epoch % cfg.checkpoint_every_epochs == 0:
            os.makedirs(work_dir, exist_ok=True)
            fp = work_dir / 'checkpoint.pt'
            agent.save(val_statistics, fp)

        if val_statistics['val/total_loss'] < total_loss:
            os.makedirs(work_dir, exist_ok=True)
            fp = work_dir / 'best_checkpoint.pt'
            agent.save(val_statistics, fp)


if __name__ == '__main__':
    main()