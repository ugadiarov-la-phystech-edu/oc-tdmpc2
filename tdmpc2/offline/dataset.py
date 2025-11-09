import glob
import warnings
from typing import NamedTuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
from torchvision.transforms import transforms
from tqdm import tqdm


class DatasetItem(NamedTuple):
    img: torch.Tensor = torch.empty(0)
    z: torch.Tensor = torch.empty(0)
    mu_scale: torch.Tensor = torch.empty(0)
    mu_depth: torch.Tensor = torch.empty(0)
    mu_features: torch.Tensor = torch.empty(0)
    obj_on: torch.Tensor = torch.empty(0)
    z_bg: torch.Tensor = torch.empty(0)
    reward: torch.Tensor = torch.empty(0)
    action: torch.Tensor = torch.empty(0)

    def to(self, device):
        return DatasetItem(*[element.to(device) for element in self])

    def update(self, **attribute2value):
        return self._replace(**attribute2value)


class DDLPFeaturesDataset(Dataset):
    def __init__(self, path, split, sample_length=1):
        assert split in ['train', 'val', 'valid']
        if split == 'valid':
            split = 'val'

        self.split_path = os.path.join(path, f'{split}.hdf5')
        self.sample_length = sample_length
        self.return_pairs = None
        self.episode_data = {}
        self.index2episode = []
        self.episode2offset = {}
        self.n_elements = 0
        with h5py.File(self.split_path, 'r') as file_obj:
            for episode_id, group in file_obj.items():
                self.episode_data[episode_id] = {key: torch.as_tensor(value[()]) for key, value in group.items()}
                if self.return_pairs is None:
                    self.return_pairs = len(self.episode_data[episode_id]['actions'].size()) < 4
                else:
                    return_pairs = len(self.episode_data[episode_id]['actions'].size()) < 4
                    assert self.return_pairs == return_pairs, f'Inconsistent data format'

                episode_len = self.episode_data[episode_id]['actions'].size()[0]
                actual_length = episode_len - self.sample_length + 1
                if actual_length <= 0:
                    warnings.warn(
                        f'Drop episode {episode_id} with length={len(episode_len)} as it too short for sample_length={self.sample_length}')
                    continue

                self.index2episode.extend([episode_id] * actual_length)
                self.episode2offset[episode_id] = self.n_elements
                self.n_elements += actual_length

    def __getitem__(self, index):
        episode_id = self.index2episode[index]
        start_index = index - self.episode2offset[episode_id]
        episode = self.episode_data[episode_id]
        k = int(self.return_pairs)
        return DatasetItem(
            z=episode['z'][start_index: start_index + self.sample_length + k],
            mu_scale=episode['mu_scale'][start_index: start_index + self.sample_length + k],
            mu_depth=episode['mu_depth'][start_index: start_index + self.sample_length + k],
            mu_features=episode['mu_features'][start_index: start_index + self.sample_length + k],
            obj_on=episode['obj_on'][start_index: start_index + self.sample_length + k],
            z_bg=episode['z_bg'][start_index: start_index + self.sample_length + k],
            reward=episode['rewards'][start_index: start_index + self.sample_length],
            action=episode['actions'][start_index: start_index + self.sample_length],
        )

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


class EpisodesSlotsDataset(Dataset):
    def __init__(self, source_root, slots_root, mode, sample_length, slots_file_name):
        assert mode in ['train', 'val', 'valid']
        if mode == 'valid':
            mode = 'val'

        self.source_root = os.path.join(source_root, mode)
        self.slots_root = os.path.join(slots_root, mode)
        self.mode = mode
        self.sample_length = sample_length
        self.slots_file_name = slots_file_name
        # Get all numbers
        self.episode_ids = sorted(os.listdir(self.source_root), key=lambda x: int(x))
        self.episode_slots = []
        self.episode_actions = []
        self.episode_rewards = []

        self._action_dim = None
        action_type = None
        min_action = np.inf
        max_action = -np.inf
        self._slot_dim = None
        for episode_id in tqdm(self.episode_ids, desc=f'Indexing split: {self.mode}'):
            slots = np.load(os.path.join(self.slots_root, episode_id, self.slots_file_name))
            if self._slot_dim is None:
                self._slot_dim = slots.shape[1:]
            else:
                assert slots.shape[1:] == self._slot_dim, \
                    f'Slot dimension mismatch. Expected: {self._slot_dim}. Actual: {slots.shape[1:]}. Episode: {episode_id}.'

            actions = np.load(os.path.join(self.source_root, episode_id, 'actions.npy'))
            if self._action_dim is None:
                self._action_dim = actions.shape[1:]
            else:
                assert actions.shape[1:] == self._action_dim, \
                    f'Action dimension mismatch. Expected: {self._action_dim}. Actual: {actions.shape[1:]}. Episode: {episode_id}.'

            if action_type is None:
                action_type = actions.dtype
            else:
                assert actions.dtype == action_type, \
                    f'Action type mismatch. Expected: {action_type}. Actual: {actions.dtype}. Episode: {episode_id}.'

            rewards = np.load(os.path.join(self.source_root, episode_id, 'rewards.npy'))
            assert len(rewards.shape) == 1, f'Expected rewards as an array of scalars. Actual: {rewards.shape}. Episode: {episode_id}.'

            assert actions.shape[0] == rewards.shape[0], \
                f'Lengths of episode actions and rewards mismatch. Actions: {actions.shape[0]}. Rewards: {rewards.shape[0]}. Episode: {episode_id}.'

            assert actions.shape[0] + 1 == slots.shape[0], \
                f'Lengths of episode actions and slots mismatch. Actions: {actions.shape[0]}. Slots: {slots.shape[0]}. Episode: {episode_id}.'

            if np.issubdtype(action_type, np.integer):
                min_action = min(min_action, actions.min())
                max_action = max(max_action, actions.max())

            self.episode_slots.append(slots)
            self.episode_actions.append(actions)
            self.episode_rewards.append(rewards)

        if np.issubdtype(action_type, np.integer):
            assert min_action == 0, \
                f'For discrete action spaces the minimal action is expected to be 0. Actual: {min_action}.'
            self.n_actions = max_action + 1
            self._action_space = 'discrete'
        else:
            self._action_space = 'continuous'

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def slot_dim(self):
        return self._slot_dim

    @property
    def action_space(self):
        return self._action_space

    def __getitem__(self, index):
        begin = np.random.choice(self.episode_slots[index].shape[0] - self.sample_length)
        z = torch.as_tensor(self.episode_slots[index][begin: begin + self.sample_length], dtype=torch.float32)
        reward = torch.as_tensor(self.episode_rewards[index][begin: begin + self.sample_length - 1], dtype=torch.float32)
        action = torch.as_tensor(self.episode_actions[index][begin: begin + self.sample_length - 1])
        if self.action_space == 'discrete':
            action = torch.nn.functional.one_hot(action, num_classes=self.n_actions)

        return DatasetItem(z=z, reward=reward, action=action.to(torch.float32))

    def __len__(self):
        return len(self.episode_slots)


if __name__ == '__main__':
    source_root = '/samsung/datasets/pick_specific_sold_policy'
    slots_root = '/samsung/datasets/pick_specific_sold_policy_savi_slots'
    mode = 'train'
    sample_length = 5
    slots_file_name = 'slots_savi.npy'
    ds = EpisodesSlotsDataset(source_root, slots_root, mode, sample_length, slots_file_name)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    for batch in dl:
        print()
