import gym
import numpy as np


class SlotExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses SlotExtractor in order to extract slots from the input image.
    """

    def __init__(self, cfg, env, slot_extractor):
        super().__init__(env)
        image_shape = (cfg.obs_size, cfg.obs_size, 3)
        assert env.observation_space.shape == image_shape, f'Expected image shape: {image_shape}. Actual image shape: {env.observation_space.shape}'

        self.cfg = cfg
        self.slot_extractor = slot_extractor
        num_slots, slot_dim = slot_extractor.get_slots_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_slots, slot_dim), dtype=np.float32
        )
        self.prev_slots = None
        self._episode_images = None

    def reset(self):
        frame = self.env.reset()
        self.prev_slots = self.slot_extractor(frame, prev_slots=None)
        self._episode_images = [frame]
        return self.prev_slots.copy()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self._episode_images.append(frame)
        prev_slots = self.prev_slots if self.cfg.pre_initialize_slots else None
        self.prev_slots = self.slot_extractor(frame, prev_slots=prev_slots)
        if done:
            info['episode_images'] = self._episode_images
            self._episode_images = None

        return self.prev_slots.copy(), reward, done, info
