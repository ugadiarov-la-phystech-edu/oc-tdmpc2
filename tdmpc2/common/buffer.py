import pickle
import random

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0
        self._buffer = None

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    @num_eps.setter
    def num_eps(self, num_eps):
        self._num_eps = num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def is_initialized(self):
        return self._buffer is not None

    def init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        storage_device = self.cfg.get('buffer_storage_device', None)
        if storage_device is None:
            mem_free, _ = torch.cuda.mem_get_info()
            bytes_per_step = sum([
                (v.numel() * v.element_size() if not isinstance(v, TensorDict) \
                     else sum([x.numel() * x.element_size() for x in v.values()])) \
                for v in tds.values()
            ]) / len(tds)
            total_bytes = bytes_per_step * self._capacity
            print(f'Storage required: {total_bytes / 1e9:.2f} GB')
            # Heuristic: decide whether to use CUDA or CPU memory
            storage_device = 'cuda' if 2.5 * total_bytes < mem_free else 'cpu'

        print(f'Using {storage_device.upper()} memory for storage.')
        buffer = self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )
        if self.cfg.get('checkpoint_buffer', None):
            print(f'Loading buffer: {self.cfg.checkpoint_buffer}')
            buffer.loads(self.cfg.checkpoint_buffer)

        self._buffer = buffer

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (arg.to(device, non_blocking=True) \
                    if arg is not None else None for arg in args)

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td['obs']
        action = td['action'][1:]
        reward = td['reward'][1:].unsqueeze(-1)
        task = td['task'][0] if 'task' in td.keys() else None
        return self._to_device(obs, action, reward, None, task)

    def add(self, td):
        """Add an episode to the buffer."""
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)

    def dumps(self, path):
        self._buffer.dumps(path)


class CompasBuffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
        self._capacity = min(self.cfg.buffer_size, self.cfg.steps)
        self._episode_observations = []
        self._episode_actions = []
        self._episode_rewards = []
        self._index2episode = []
        self._episode2offset = [0]
        self._batch_size = self.cfg.batch_size
        self._horizon = self.cfg.horizon
        self._context_len = self.cfg.compas_max_timestep
        self._buffer = None

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return len(self._episode_observations)

    @num_eps.setter
    def num_eps(self, num_eps):
        return

    def is_initialized(self):
        return True

    def init(self, tds):
        return

    def add(self, td):
        """Add an episode to the buffer."""
        self._episode_observations.append(td['obs'].detach().cpu())
        self._episode_actions.append(td['action'].detach().cpu())
        self._episode_rewards.append(td['reward'].detach().cpu())

        effective_episode_len = td['obs'].size()[0] - self._horizon
        self._index2episode.extend([self.num_eps - 1] * effective_episode_len)
        self._episode2offset.append(self._episode2offset[-1] + effective_episode_len)

        return self.num_eps

    @staticmethod
    def _pad_left(tensor, pad_len):
        if pad_len == 0:
            return tensor

        pad = torch.empty(pad_len, *tensor.size()[1:], dtype=tensor.dtype, device=tensor.device)
        return torch.cat([pad, tensor])

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        size = len(self._index2episode)
        if size >= self._batch_size * (self._horizon + 1):
            absolute_indices = random.sample(range(size), k=self._batch_size)
        else:
            absolute_indices = random.choices(range(size), k=self._batch_size)

        observations = []
        actions = []
        rewards = []
        pads = []
        for absolute_index in absolute_indices:
            episode_id = self._index2episode[absolute_index]
            in_episode_index = absolute_index - self._episode2offset[episode_id] - self._context_len
            pad_len = 0
            if in_episode_index < 0:
                pad_len = -in_episode_index
                in_episode_index = 0

            slc = slice(in_episode_index, in_episode_index + self._horizon + 1 + self._context_len - pad_len)
            obs = self._pad_left(self._episode_observations[episode_id][slc], pad_len)
            action = self._pad_left(self._episode_actions[episode_id][slc], pad_len)
            reward = self._pad_left(self._episode_rewards[episode_id][slc], pad_len)

            observations.append(obs)
            actions.append(action[1:])
            rewards.append(reward[1:])
            pads.append(pad_len)

        observations = torch.stack(observations).to(self._device).movedim(0, 1)
        actions = torch.stack(actions).to(self._device).movedim(0, 1)
        rewards = torch.stack(rewards).to(self._device).movedim(0, 1)
        pads = torch.as_tensor(np.asarray(pads, dtype=np.int64), device=self._device)

        return observations, actions, rewards.unsqueeze(-1), pads, None

    def dumps(self, path):
        with open(path, 'wb') as file_obj:
            pickle.dump(self, file_obj)
