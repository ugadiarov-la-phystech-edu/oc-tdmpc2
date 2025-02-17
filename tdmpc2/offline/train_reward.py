import argparse
import collections
import itertools
import time
from typing import Tuple

import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import layers
from common.layers import mlp, enc
from common.world_model import OCRewardModel
from dlp.policies import EITCritic
from offline.dataset import DDLPFeaturesDataset, DatasetItem, EpisodesDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def foreground_features(x: DatasetItem):
    return torch.cat([x.z, x.mu_scale, x.mu_depth, x.mu_features, x.obj_on], dim=-1)


def create_dataloader(dataset_type, dataset_path, split, batch_size, num_workers, **kwargs):
    if dataset_type == 'ddlp':
        dataset = DDLPFeaturesDataset(dataset_path, split, **kwargs)
    elif dataset_type == 'rgb':
        dataset = EpisodesDataset(dataset_path, split, **kwargs)
    else:
        raise ValueError(f'Unexpected dataset type: {dataset_type}')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=num_workers)
    return dataloader


class GNNRewardModel(nn.Module):
    def __init__(self, config, background_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.background_projection = None
        n_slots = self.config.n_slots
        if background_dim is not None:
            self.background_projection = mlp(background_dim, [], self.config.slot_dim)
            n_slots += 1

        self.reward_model = OCRewardModel(config, n_slots=n_slots)

    def forward(self, fg, bg, action):
        x = fg
        if self.background_projection is not None:
            background_particle = self.background_projection(bg)
            x = torch.cat([x, background_particle], dim=1)

        return self.reward_model(x, action)


class EITRewardModel(EITCritic):
    def __init__(self, cfg, particle_fdim, action_dim, background_fdim):
        super().__init__(cfg, particle_fdim, action_dim, 1, 1, background_fdim)
        self.use_background = cfg.eit_use_background

    def forward(self, fg, bg, action) -> Tuple[torch.Tensor, ...]:
        if self.use_background:
            bg = bg.squeeze(1)
        else:
            bg = None

        return super().forward(fg, action, bg)[0]


class MonolithicRewardModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._encoder = enc(cfg)
        latent_dim = self._encoder[cfg.obs](torch.zeros(1, *cfg.obs_shape[cfg.obs], dtype=torch.float32)).size()[1]
        self._reward = layers.mlp(latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)

    def forward(self, x: DatasetItem):
        obs = x.img.flatten(start_dim=1, end_dim=2)
        action = x.action[:, 0]
        return self._reward(torch.cat([self._encoder[self.cfg.obs](obs), action], dim=-1))


def run(reward_model: nn.Module, dataloader: DataLoader, device: str, is_train: bool = True, reward_discount=1.0):
    if is_train:
        mode = 'train'
        reward_model.train()
        reward_model.requires_grad_(True)
    else:
        mode = 'val'
        reward_model.eval()
        reward_model.requires_grad_(False)

    pbar = tqdm(iterable=dataloader)
    metrics = collections.defaultdict(list)
    for batch in pbar:
        batch = batch.to(device)
        if len(batch.action.size()) == 5:
            sample_length = batch.action.size()[1]
            assert sample_length == 1, f'Expected: sample_length == 1. Actual: sample_length={sample_length}'
            batch_size, _, prediction_horizon = batch.action.size()[:3]
            fg = foreground_features(batch)[:, 0].movedim(3, 2).flatten(start_dim=3).flatten(end_dim=1)
            bg = batch.z_bg[:, 0].movedim(3, 2).flatten(start_dim=3).flatten(end_dim=1)
            predicted_rewards = reward_model(fg, bg, batch.action[:, 0, :, -1].flatten(end_dim=1)).reshape(batch_size, prediction_horizon)
            gt_rewards = batch.reward[:, 0].reshape(batch_size * prediction_horizon, -1).reshape(batch_size, prediction_horizon)
            discount = torch.pow(reward_discount * torch.ones(prediction_horizon, dtype=torch.float32, device=device),
                                 torch.arange(prediction_horizon, dtype=torch.float32, device=device))
            loss = torch.mean((predicted_rewards - gt_rewards) ** 2, dim=0)
            for step, step_loss in enumerate(loss):
                metrics[f'loss_step-{step}'].append(step_loss.item())

            loss = torch.mean(discount * loss)
        else:
            fg = foreground_features(batch)[:, :-1].permute(0, 3, 1, 2, 4).flatten(start_dim=2)
            bg = batch.z_bg[:, :-1].permute(0, 3, 1, 2, 4).flatten(start_dim=2)
            predicted_rewards = reward_model(fg, bg, batch.action[:, -1])
            gt_rewards = batch.reward[:, -1:]
            loss = nn.functional.mse_loss(predicted_rewards, gt_rewards)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description_str(f'{mode} epoch #{epoch}')
        pbar.set_postfix(loss=loss.item())
        metrics['loss'].append(loss.item())

    pbar.close()

    return {key: sum(value) / len(value) for key, value in metrics.items()}


def add_prefix(dictionary, prefix):
    return {prefix + key: value for key, value in dictionary.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--use_interactions', type=str2bool, default=True)
    parser.add_argument('--use_background', type=str2bool, required=True)
    parser.add_argument('--eit_use_masking', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['gnn', 'eit', 'monolithic'], required=True)
    parser.add_argument('--sample_length', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_every_hours', type=float, default=3)
    parser.add_argument('--reward_discount', type=float, default=1.0)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    if args.model_type == 'monolithic':
        dataset_type = 'rgb'
        kwargs = dict(sample_length=3, res=64, episodic_on_train=False, episodic_on_val=False, use_actions=True,
                      duplicate_on_episode_start=True)
    else:
        dataset_type = 'ddlp'
        kwargs = {'sample_length': args.sample_length}

    train_dataloader = create_dataloader(dataset_type, args.dataset_path, 'train', args.batch_size, args.num_workers,
                                         **kwargs)
    val_dataloader = create_dataloader(dataset_type, args.dataset_path, 'val', args.batch_size, args.num_workers,
                                       **kwargs)

    # action.shape -> batch_size, sample_length, action_dim
    # z.shape -> batch_size, sample_length + 1, timestep_horizon, n_particles, 2
    sample = next(iter(train_dataloader))
    action_dim = sample.action[0].size()[-1]
    config = {'latent_dim': args.latent_dim, 'action_dim': action_dim}
    if args.model_type == 'monolithic':
        num_frames = sample.img[0].size()[0]
        config.update({'obs': 'rgb', 'obs_shape': {'rgb': (3 * num_frames, 64, 64)}, 'task_dim': 0, 'num_enc_layers': 2, 'enc_dim': 256,
                       'num_channels': 32, 'simnorm_dim': 8, 'mlp_dim': 512})
    else:
        n_slots = sample.z.size()[-2]
        slot_dim = foreground_features(sample).size()[-1] * args.sample_length * sample.z.size()[-3]
        background_slot_dim = None
        if args.use_background:
            background_slot_dim = sample.z_bg.size()[-1] * args.sample_length * sample.z.size()[-3]

        config = {'latent_dim': args.latent_dim, 'action_dim': action_dim, 'use_interactions': args.use_interactions,
                      'n_slots': n_slots, 'slot_dim': slot_dim, 'num_bins': 1}

    config = OmegaConf.create(config)
    if args.model_type == 'gnn':
        reward_model = GNNRewardModel(config, background_slot_dim)
    elif args.model_type == 'eit':
        config['eit_embed_dim'] = 64
        config['eit_h_dim'] = 256
        config['eit_n_head'] = 8
        config['eit_dropout'] = 0
        config['eit_action_particle'] = True
        config['eit_masking'] = True
        config['eit_use_background'] = args.use_background
        reward_model = EITRewardModel(config, slot_dim, action_dim, background_slot_dim)
    elif args.model_type == 'monolithic':
        reward_model = MonolithicRewardModel(config)
    else:
        raise ValueError(f'Unexpected model type: {args.model_type}')

    reward_model = reward_model.to(args.device)

    save_time = time.time() + args.save_every_hours * 60 * 60
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=args.lr,)
    for epoch in itertools.count():
        train_metrics = run(reward_model, train_dataloader, args.device, is_train=True, reward_discount=args.reward_discount)
        val_metrics = run(reward_model, val_dataloader, args.device, is_train=False, reward_discount=args.reward_discount)
        if args.wandb_project:
            if wandb.run is None and args.wandb_project is not None:
                wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run, resume='never',
                           config={**vars(args), **OmegaConf.to_container(config)})

            if wandb.run is not None:
                record = {'epoch': epoch}
                record.update(add_prefix(train_metrics, prefix='train/'))
                record.update(add_prefix(val_metrics, prefix='val/'))
                wandb.log(record)

        if time.time() > save_time:
            torch.save({'epoch': epoch, 'model_state_dict': reward_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, args.checkpoint_path)
            save_time = time.time() + args.save_every_hours * 60 * 60

    if args.wandb_project:
        wandb.finish()
