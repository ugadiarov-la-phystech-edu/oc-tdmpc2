import argparse
import itertools

import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.layers import mlp
from common.world_model import OCRewardModel
from offline.dataset import DDLPFeaturesDataset, DatasetItem


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_dataloader(dataset_path, split, batch_size, num_workers):
    dataset = DDLPFeaturesDataset(dataset_path, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=num_workers)
    return dataloader


class RewardModel(nn.Module):
    def __init__(self, config, background_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.background_projection = None
        n_slots = self.config.n_slots
        if background_dim is not None:
            self.background_projection = mlp(background_dim, [], self.config.slot_dim)
            n_slots += 1

        self.reward_model = OCRewardModel(config, n_slots=n_slots)

    def forward(self, x: DatasetItem):
        if self.background_projection is not None:
            background_particle = self.background_projection(x.bg)
            x = x.update(fg=torch.cat([x.fg, background_particle.unsqueeze(1)], dim=1))

        return self.reward_model(x.fg, x.action)


def run(reward_model: nn.Module, dataloader: DataLoader, device: str, is_train: bool = True):
    if is_train:
        mode = 'train'
        reward_model.train()
        reward_model.requires_grad_(True)
    else:
        mode = 'val'
        reward_model.eval()
        reward_model.requires_grad_(False)

    pbar = tqdm(iterable=dataloader)
    losses = []
    for batch in pbar:
        batch = batch.to(device)
        predicted_rewards = reward_model(batch)
        loss = nn.functional.mse_loss(predicted_rewards.reshape_as(batch.reward), batch.reward)
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
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    train_dataloader = create_dataloader(args.dataset_path, 'train', args.batch_size, args.num_workers)
    val_dataloader = create_dataloader(args.dataset_path, 'val', args.batch_size, args.num_workers)

    sample = next(iter(train_dataloader))
    n_slots, slot_dim = sample.fg[0].size()
    action_dim = sample.action[0].size()[0]

    config = OmegaConf.create(
        {'latent_dim': args.latent_dim, 'use_interactions': args.use_interactions, 'num_bins': 1, 'n_slots': n_slots,
         'slot_dim': slot_dim, 'action_dim': action_dim})

    background_dim = sample.bg[0].size()[0] if args.use_background else None
    reward_model = RewardModel(config, background_dim).to(args.device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=args.lr,)
    for epoch in itertools.count():
        train_loss = run(reward_model, train_dataloader, args.device, is_train=True)
        val_loss = run(reward_model, val_dataloader, args.device, is_train=False)
        if args.wandb_project:
            if wandb.run is None:
                wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run, resume='never',
                           config=vars(args))

            wandb.log({'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss})

    if args.wandb_project:
        wandb.finish()
