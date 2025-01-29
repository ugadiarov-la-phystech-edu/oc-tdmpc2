import argparse
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
from common.world_model import OCRewardModel, OCDynamicsModel
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


def create_dataloader(dataset_type, dataset_path, split, batch_size, num_workers, **kwargs):
    if dataset_type == 'ddlp':
        dataset = DDLPFeaturesDataset(dataset_path, split, **kwargs)
    elif dataset_type == 'rgb':
        dataset = EpisodesDataset(dataset_path, split, **kwargs)
    else:
        raise ValueError(f'Unexpected dataset type: {dataset_type}')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=num_workers)
    return dataloader


def foreground_features(x: DatasetItem):
    return torch.cat([x.z, x.mu_scale, x.mu_depth, x.mu_features, x.obj_on], dim=-1)


class GNNTransitionModel(nn.Module):
    def __init__(self, config, output_dim, background_slot_dim=None, background_output_dim=None):
        super().__init__()
        self.config = config
        self.background_projection = None
        n_slots = self.config.n_slots
        if background_slot_dim is not None:
            self.background_projection = mlp(background_slot_dim, [], slot_dim)
            self.background_reconstruction = mlp(output_dim, [], background_output_dim)
            n_slots += 1

        self.model = OCDynamicsModel(config, n_slots=n_slots, output_dim=output_dim)

    def forward(self, fg, bg, action):
        x = fg
        if self.background_projection is not None:
            background_particle = self.background_projection(bg)
            x = torch.cat([x, background_particle], dim=1)

        output = self.model(x, action)
        pred_fg = output
        pred_bg = None
        if self.background_reconstruction:
            pred_fg = output[:, :-1]
            pred_bg = self.background_reconstruction(output[:, -1:])

        return pred_fg, pred_bg


def run(model: nn.Module, dataloader: DataLoader, device: str, use_background: bool, is_train: bool = True):
    if is_train:
        mode = 'train'
        model.train()
        model.requires_grad_(True)
    else:
        mode = 'val'
        model.eval()
        model.requires_grad_(False)

    pbar = tqdm(iterable=dataloader)
    losses = []
    for batch in pbar:
        batch = batch.to(device)
        fg = foreground_features(batch)
        bg = batch.z_bg
        next_fg = fg[:, -1]
        next_bg = bg[:, -1]
        fg = fg[:, :-1].permute(0, 2, 1, 3).flatten(start_dim=2)
        bg = bg[:, :-1].permute(0, 2, 1, 3).flatten(start_dim=2)
        pred_fg, pred_bg = model(fg, bg, batch.action[:, -1])
        if use_background:
            loss = nn.functional.mse_loss(pred_fg, next_fg, reduction='sum')
            loss += nn.functional.mse_loss(pred_bg, next_bg, reduction='sum')
            loss /= next_fg.numel() + next_bg.numel()
        else:
            loss = nn.functional.mse_loss(pred_fg, next_fg)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description_str(f'{mode} epoch #{epoch}')
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    pbar.close()

    return sum(losses) / len(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--use_interactions', type=str2bool, default=True)
    parser.add_argument('--use_background', type=str2bool, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_run', type=str, default=None)
    parser.add_argument('--sample_length', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_every_hours', type=int, default=3)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    kwargs = {'sample_length': args.sample_length}
    train_dataloader = create_dataloader('ddlp', args.dataset_path, 'train', args.batch_size, args.num_workers,
                                         **kwargs)
    val_dataloader = create_dataloader('ddlp', args.dataset_path, 'val', args.batch_size, args.num_workers,
                                       **kwargs)

    sample = next(iter(train_dataloader))
    action_dim = sample.action[0].size()[-1]
    n_slots = sample.z.size()[-2]
    output_dim = foreground_features(sample).size()[-1]
    slot_dim = output_dim * args.sample_length
    background_output_dim = None
    background_slot_dim = None
    if args.use_background:
        background_output_dim = sample.z_bg.size()[-1]
        background_slot_dim = background_output_dim * args.sample_length

    config = {'latent_dim': args.latent_dim, 'action_dim': action_dim, 'use_interactions': args.use_interactions, 'n_slots': n_slots,
         'slot_dim': slot_dim,}

    config = OmegaConf.create(config)
    transition_model = GNNTransitionModel(config, output_dim=output_dim, background_slot_dim=background_slot_dim,
                                          background_output_dim=background_output_dim).to(args.device)

    optimizer = torch.optim.Adam(transition_model.parameters(), lr=args.lr,)
    save_time = time.time() + args.save_every_hours * 60 * 60
    for epoch in itertools.count():
        train_loss = run(transition_model, train_dataloader, args.device, use_background=args.use_background, is_train=True)
        val_loss = run(transition_model, val_dataloader, args.device, use_background=args.use_background, is_train=False)
        if args.wandb_project:
            if wandb.run is None and args.wandb_project is not None:
                wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run, resume='never',
                           config={**vars(args), **OmegaConf.to_container(config)})

            if wandb.run is not None:
                wandb.log({'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss})

        if time.time() > save_time:
            torch.save({'epoch': epoch, 'model_state_dict': transition_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, args.checkpoint_path)
            save_time = time.time() + args.save_every_hours * 60 * 60

    if args.wandb_project:
        wandb.finish()
