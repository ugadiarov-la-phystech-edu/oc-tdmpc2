import glob
import warnings
from typing import NamedTuple

import h5py
import numpy as np
from PIL import Image
from tensordict import TensorDict
from torch.utils.data import Dataset
import os
import torch
from torchvision.transforms import transforms


class DatasetItem(NamedTuple):
    img: torch.Tensor = torch.empty(0)
    fg: torch.Tensor = torch.empty(0)
    bg: torch.Tensor = torch.empty(0)
    action: torch.Tensor = torch.empty(0)
    reward: torch.Tensor = torch.empty(0)

    def to(self, device):
        return DatasetItem(*[element.to(device) for element in self])

    def update(self, **attribute2value):
        return self._replace(**attribute2value)


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
            element['fg'] = element.pop('fg_representation').permute((1, 0, 2)).flatten(start_dim=-2)
            element['bg'] = element.pop('bg_representation').flatten(start_dim=-2)
            element['action'] = element.pop('actions').flatten(start_dim=-2)
            element['reward'] = element.pop('rewards')

        return DatasetItem(**element)

    def __len__(self):
        return self.n_elements


class EpisodesDataset(Dataset):
    def __init__(self, root, mode, sample_length=1, res=128, episodic_on_train=False, episodic_on_val=False,
                 use_actions=False, duplicate_on_episode_start=False):
        assert mode in ['train', 'val', 'valid']
        if mode == 'valid':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.res = res
        self.use_actions = use_actions

        self.mode = mode
        self.episodic = (self.mode == 'train' and episodic_on_train) or (self.mode == 'val' and episodic_on_val)
        self.sample_length = sample_length
        self.duplicate_on_episode_start = duplicate_on_episode_start
        assert sample_length > 1 or not duplicate_on_episode_start, 'Duplication can be used only for video'

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue

        self.folders.sort(key=lambda x: int(x))

        self.episode_images = []
        self.episode2offset = [0]
        self.index2episode = []
        self.actions = []
        self.rewards = []
        action_dim = None
        action_type = None
        min_action = np.inf
        max_action = -np.inf
        for i, f in enumerate(self.folders):
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(os.path.join(dir_name, '*.png')))
            actual_length = len(paths) - self.sample_length + 1
            if actual_length <= 0:
                warnings.warn(
                    f'Drop episode {dir_name} with length={len(paths)} as it too short for sample_length={self.sample_length}')
                continue

            if self.duplicate_on_episode_start:
                actual_length = len(paths) - 1

            self.rewards.append(np.load(os.path.join(dir_name, 'rewards.npy')))
            if self.use_actions:
                actions_path = os.path.join(dir_name, 'actions.npy')
                assert os.path.exists(actions_path), f'{os.path.abspath(actions_path)} does not exists.'
                episode_actions = np.load(actions_path)
                self.actions.append(episode_actions)
                if action_dim is None:
                    action_dim = episode_actions.shape[1:]
                else:
                    assert episode_actions.shape[1:] == action_dim, \
                        f'Action dimension mismatch. Expected: {action_dim}. Actual: {episode_actions.shape[1:]}.'

                if action_type is None:
                    action_type = episode_actions.dtype
                else:
                    assert episode_actions.dtype == action_type, \
                        f'Action type mismatch. Expected: {action_type}. Actual: {episode_actions.dtype}.'

                if np.issubdtype(action_type, np.integer):
                    min_action = min(min_action, episode_actions.min())
                    max_action = max(max_action, episode_actions.max())

            get_num = lambda x: int(os.path.splitext(os.path.basename(x))[0])
            paths.sort(key=get_num)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

        if np.issubdtype(action_type, np.integer):
            assert min_action == 0, \
                f'For discrete action spaces the minimal action is expected to be 0. Actual: {min_action}.'
            self.n_actions = max_action + 1
            self.action_space = 'discrete'
        else:
            self.action_space = 'continuous'

    def __getitem__(self, index):
        if self.episodic:
            ep = index
            begin = 0
            end = len(self.episode_images[ep])
        else:
            ep = self.index2episode[index]
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            if self.duplicate_on_episode_start:
                end = (index + 1) - offset + 1
                begin = end - self.sample_length
            else:
                begin = index - offset
                end = begin + self.sample_length

        if self.use_actions:
            actual_begin = max(begin, 0)
            action = torch.as_tensor(self.actions[ep][actual_begin: end - 1])
            if self.action_space == 'discrete':
                action = torch.nn.functional.one_hot(action, num_classes=self.n_actions)

            action = action.to(torch.float32)
            if actual_begin != begin:
                assert self.duplicate_on_episode_start
                empty_action = torch.zeros(size=(actual_begin - begin, *action.size()[1:]), dtype=action.dtype)
                action = torch.cat([empty_action, action], dim=0)
        else:
            action = torch.zeros(0)

        reward = torch.as_tensor(self.rewards[ep][end - 2], dtype=torch.float32)
        revered_sequence_images = []
        for image_index in reversed(range(begin, end)):
            if image_index < 0:
                assert self.duplicate_on_episode_start
                revered_sequence_images.append(revered_sequence_images[-1])
            else:
                img = Image.open(self.episode_images[ep][image_index])
                img = img.resize((self.res, self.res))
                img = transforms.ToTensor()(img)[:3]
                revered_sequence_images.append(img)

        return DatasetItem(img=torch.stack(revered_sequence_images[::-1], dim=0).float(), action=action, reward=reward)

    def __len__(self):
        if self.episodic:
            # Number of episodes
            return len(self.episode_images)
        else:
            # Number of available sequences of length self.sample_length
            return len(self.index2episode)


if __name__ == '__main__':
    ds = DDLPFeaturesDataset(path='data', split='val')
    print('Length:', len(ds))
    for i in range(len(ds)):
        ds[i]

    ds = EpisodesDataset(root='/media/elfray/hdd_ext4/projects/ddlp/datasets/cw_reaching-hard', mode='val', sample_length=3, res=64, episodic_on_train=False,
                         episodic_on_val=False, use_actions=True, duplicate_on_episode_start=True)
    print('Length:', len(ds))
    for i in range(len(ds)):
        ds[i]
