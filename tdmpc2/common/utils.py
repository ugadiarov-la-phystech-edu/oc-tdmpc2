"""Utility functions."""
import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname, use_rle, image_shape):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('use_rle', data=np.array(use_rle))
        hf.create_dataset('image_shape', data=np.array(image_shape))
        for episode in range(len(array_dict)):
            grp = hf.create_group(str(episode))
            for array_name, array_value in array_dict[episode].items():
                if use_rle and isinstance(array_value[0], np.ndarray):
                    # align sizes of rle arrays
                    max_len = len(max(array_value, key=len))
                    for j in range(len(array_value)):
                        element = array_value[j]
                        array_value[j] = np.pad(element, pad_width=(max_len - len(element), 0), constant_values=0)

                grp.create_dataset(array_name, data=array_value)


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        use_rle = 'use_rle' in hf and np.asarray(hf['use_rle']).item()
        image_shape = np.asarray(hf['image_shape']) if 'image_shape' in hf else None
        i = 0
        for grp in hf.keys():
            if grp in ('use_rle', 'image_shape'):
                continue

            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]

            i += 1

    return array_dict, use_rle, image_shape


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, hdf5_file_auxiliary=None, gamma=1):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.gamma = gamma
        self.has_returns = False
        self.experience_buffer, self.use_rle, self.image_shape = load_list_dict_h5py(hdf5_file)
        self.n_boxes = -1
        if 'moving_boxes' in self.experience_buffer[0]:
            self.n_boxes = len(self.experience_buffer[0]['moving_boxes'][0])

        self.action_dim = 0
        episode_actions_shape = self.experience_buffer[0]['action'].shape
        if len(episode_actions_shape) > 1:
            assert len(
                episode_actions_shape) == 2, f'Expected flatten actions, actual action shape: {episode_actions_shape[1:]}'
            self.action_dim = self.experience_buffer[0]['action'].shape[1]

        if hdf5_file_auxiliary is not None:
            experience_buffer_states, use_rle_states, image_shape_states = load_list_dict_h5py(hdf5_file_auxiliary)
            n_boxes_states = -1
            if 'moving_boxes' in experience_buffer_states[0]:
                n_boxes_states = len(experience_buffer_states[0]['moving_boxes'][0])

            assert use_rle_states == self.use_rle
            assert np.array_equal(image_shape_states, self.image_shape)
            assert n_boxes_states == self.n_boxes

            self.experience_buffer += experience_buffer_states

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action']) + 1
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            self.experience_buffer[ep]['return'] = np.full(shape=(num_steps,), fill_value=np.nan, dtype=np.float32)
            step += num_steps

        self.num_steps = step
        print(f'Dataset length: {self.num_steps}\n')

    def compute_returns(self):
        for episode in self.experience_buffer:
            returns = [np.zeros_like(episode['reward'][0])]
            for reward in episode['reward'][::-1]:
                returns.append(reward + self.gamma * returns[-1])

            episode['return'] = np.asarray(returns[::-1])

        self.has_returns = True

    def __len__(self):
        return self.num_steps

    def _get_observation(self, ep, step, next_obs=False):
        obs_key = 'next_obs' if next_obs else 'obs'
        if not self.use_rle:
            return to_float(self.experience_buffer[ep][obs_key][step])

        starts = self.experience_buffer[ep][f'{obs_key}_starts'][step]
        lengths = self.experience_buffer[ep][f'{obs_key}_lengths'][step]
        values = self.experience_buffer[ep][f'{obs_key}_values'][step]
        return to_float(rldecode(starts, lengths, values).reshape(self.image_shape))

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        if step == len(self.experience_buffer[ep]['action']):
            # Get the terminal observation
            obs = self._get_observation(ep, step - 1, next_obs=True)
            if self.action_dim == 0:
                action = -1
            else:
                action = np.full((self.action_dim,), fill_value=-1, dtype=np.float32)

            if self.n_boxes == -1:
                moving_boxes = -1
            else:
                moving_boxes = np.full((self.n_boxes,), fill_value=-1, dtype=np.int64)
            next_obs = np.full_like(obs, fill_value=0)
            reward = np.nan
            is_terminal = True
        else:
            obs = self._get_observation(ep, step)
            action = self.experience_buffer[ep]['action'][step]
            if self.n_boxes == -1:
                moving_boxes = -1
            else:
                moving_boxes = self.experience_buffer[ep]['moving_boxes'][step]
            next_obs = self._get_observation(ep, step, next_obs=True)
            reward = self.experience_buffer[ep]['reward'][step]
            is_terminal = False

        returns = self.experience_buffer[ep]['return'][step]

        return obs, action, moving_boxes, next_obs, reward, returns, is_terminal


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer, self.use_rle, self.image_shape = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def _get_observation(self, ep, step, next_obs=False):
        obs_key = 'next_obs' if next_obs else 'obs'
        if not self.use_rle:
            return to_float(self.experience_buffer[ep][obs_key][step])

        starts = self.experience_buffer[ep][f'{obs_key}_starts'][step]
        lengths = self.experience_buffer[ep][f'{obs_key}_lengths'][step]
        values = self.experience_buffer[ep][f'{obs_key}_values'][step]
        return to_float(rldecode(starts, lengths, values).reshape(self.image_shape))

    def __getitem__(self, idx):
        observations = []
        actions = []
        rewards = []
        for i in range(self.path_length):
            obs = self._get_observation(idx, i)
            action = self.experience_buffer[idx]['action'][i]
            reward = self.experience_buffer[idx]['reward'][i]
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
        obs = self._get_observation(idx, self.path_length - 1)
        observations.append(obs)
        return observations, actions, rewards


def observed_colors(num_colors, mode, randomize=True):
    if mode == 'ZeroShot':
        c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
    else:
        c = (np.arange(num_colors)) / (num_colors - 1)
        if not randomize:
            return c

        diff = 1.0 / (num_colors - 1)
        if mode == 'Train':
            diff = diff / 8.0
        elif mode == 'Test-v1':
            diff = diff / 4.0
        elif mode == 'Test-v2':
            diff = diff / 3.0
        elif mode == 'Test-v3':
            diff = diff / 2.0

        unif = np.random.uniform(-diff + EPS, diff - EPS, size=num_colors)
        unif[0] = abs(unif[0])
        unif[-1] = -abs(unif[-1])

        c = c + unif

    return c


def get_cmap(cmap, mode):
    length = 9
    if cmap == 'Sets':
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Set1')
        else:
            cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
            length = [9, 12]
    else:
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Pastel1')
        else:
            cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
            length = [9, 8]

    return cmap, length


def unobserved_colors(cmap, num_colors, mode, new_colors=None):
    if mode in ['Train', 'ZeroShotShape']:
        cm, length = get_cmap(cmap, mode)
        weights = np.sort(np.random.choice(length, num_colors, replace=False))
        colors = [cm(i / length) for i in weights]
    else:
        cm, length = get_cmap(cmap, mode)
        cm1, cm2 = cm
        length1, length2 = length
        l = length1 + len(new_colors)
        w = np.sort(np.random.choice(l, num_colors, replace=False))
        colors = []
        weights = []
        for i in w:
            if i < length1:
                colors.append(cm1(i / length1))
                weights.append(i)
            else:
                colors.append(cm2(new_colors[i - length1] / length2))
                weights.append(new_colors[i - length1] + 0.5)

    return colors, weights


def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True,
                           mode='Train', new_colors=None, randomize=True):
    """Get color array from matplotlib colormap."""
    if observed:
        c = observed_colors(num_colors, mode, randomize=randomize)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx
                   for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)

    return colors, weights


# https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
def rlencode(x, dropna=False, index_dtype=np.int32):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    index_dtype: np.dtype, optional
        dtype for 'start positions' and 'run lengths' arrays

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=index_dtype),
                np.array([], dtype=index_dtype),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1].astype(index_dtype)
    lengths = np.diff(np.r_[starts, n]).astype(index_dtype)
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values


def rldecode(starts, lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.

    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.

    Returns
    -------
    1D array. Missing data will be filled with NaNs.

    """
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    # TODO: check validity of rle
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.full(n, np.nan, dtype=values.dtype)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x


def make_node_mlp_layers(num_layers, input_dim, hidden_dim, output_dim, act_fn, layer_norm):
    layers = []

    for idx in range(num_layers):

        if idx == 0:
            # first layer, input_dim => hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 2:
            # layer before the last, add layer norm
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 1:
            # last layer, hidden_dim => output_dim and no activation
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # all other layers, hidden_dim => hidden_dim
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))

    return layers
