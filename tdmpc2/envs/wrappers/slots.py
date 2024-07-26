import gym
import numpy as np


class SlotExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses SlotExtractor in order to extract slots from the input image.
    """

    def __init__(self, cfg, env, slot_extractor):
        super().__init__(env)
        image_shape = (224, 224, 3)
        assert env.observation_space.shape == image_shape, f'Expected image shape: {image_shape}. Actual image shape: {env.observation_space.shape}'

        self.cfg = cfg
        self.slot_extractor = slot_extractor
        num_slots, slot_dim = slot_extractor.get_slots_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_slots, slot_dim), dtype=np.float32
        )
        self.prev_slots = None

    def _get_slots(self, frame, prev_slots=None):
        if prev_slots is None:
            prev_slots = self.slot_extractor(frame, prev_slots=None)

        return self.slot_extractor(frame, prev_slots=prev_slots)

    def reset(self):
        frame = self.env.reset()
        self.prev_slots = self._get_slots(frame, prev_slots=None)
        return self.prev_slots.copy()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.prev_slots = self._get_slots(frame, prev_slots=self.prev_slots)
        return self.prev_slots.copy(), reward, done, info
