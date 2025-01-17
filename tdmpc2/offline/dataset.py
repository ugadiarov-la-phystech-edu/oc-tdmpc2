from typing import NamedTuple

import h5py
from tensordict import TensorDict
from torch.utils.data import Dataset
import os
import torch


class DatasetItem(NamedTuple):
    fg: torch.Tensor
    bg: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor

    def to(self, device):
        return DatasetItem(*[element.to(device) for element in self])

    def update(self, fg=None, bg=None, action=None, reward=None):
        args = list(locals().values())[1:]
        return DatasetItem(*[x if y is None else y for x, y in zip(self, args)])


class DDLPFeaturesDataset(Dataset):
    def __init__(self, path, split, do_flatten_over_timestep_horizon=True):
        assert split in ['train', 'val', 'valid']
        if split == 'valid':
            split = 'val'

        self.do_flatten_over_timestep_horizon = do_flatten_over_timestep_horizon
        self.split_path = os.path.join(path, f'{split}.hdf5')
        self.episode_data = {}
        self.index2episode = []
        self.episode2offset = {}
        self.n_elements = 0
        with h5py.File(self.split_path, 'r') as file_obj:
            for episode_id, group in file_obj.items():
                self.episode_data[episode_id] = TensorDict({key: torch.as_tensor(group[key][()]) for key in (
                'fg_representation', 'bg_representation', 'actions', 'rewards')}, batch_size=group['actions'].shape[0])
                episode_len = self.episode_data[episode_id]['actions'].size()[0]
                self.index2episode.extend([episode_id] * episode_len)
                self.episode2offset[episode_id] = self.n_elements
                self.n_elements += episode_len

    def __getitem__(self, index):
        episode_id = self.index2episode[index]
        element = self.episode_data[episode_id][index - self.episode2offset[episode_id]]
        if self.do_flatten_over_timestep_horizon:
            element['fg_representation'] = element['fg_representation'].permute((1, 0, 2)).flatten(start_dim=-2)
            element['bg_representation'] = element['bg_representation'].flatten(start_dim=-2)
            element['actions'] = element['actions'].flatten(start_dim=-2)

        return DatasetItem(*[element[key] for key in ('fg_representation', 'bg_representation', 'actions', 'rewards')])

    def __len__(self):
        return self.n_elements


if __name__ == '__main__':
    ds = DDLPFeaturesDataset(path='data', split='val')
    print('Length:', len(ds))
    for i in range(len(ds)):
        ds[i]
