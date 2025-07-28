import gym
import numpy as np
import torch


class FeaturesWrapper(gym.Wrapper):
    """
    Wrapper uses FeaturesExtractor in order to extract features from the input image.
    """

    def __init__(self, cfg, env, features_extractor):
        super().__init__(env)
        image_shape = (cfg.obs_size, cfg.obs_size, 3)
        assert env.observation_space.shape == image_shape, f'Expected image shape: {image_shape}. Actual image shape: {env.observation_space.shape}'

        self.cfg = cfg
        self._features_extractor = features_extractor
        self._device = next(self._features_extractor.parameters()).device
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.ch * self.cfg.num_frames,), dtype=np.float32
        )
        self._history = np.zeros(shape=(self.cfg.num_frames, self.cfg.ch), dtype=np.float32)
        self._episode_images = None

    def _to_features(self, frame):
        frame = torch.as_tensor(frame / 255., dtype=torch.float32, device=self._device).movedim(-1, 0).unsqueeze_(0)
        features = self._features_extractor(frame).detach().cpu().numpy()

        return features

    def reset(self):
        frame = self.env.reset()
        self._episode_images = [frame]
        features = self._to_features(frame)
        self._history[:] = features[np.newaxis]
        return self._history.reshape(-1).copy()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self._episode_images.append(frame)
        self._history[:-1] = self._history[1:]
        self._history[-1] = self._to_features(frame)
        if done:
            info['episode_images'] = self._episode_images
            self._episode_images = None

        return self._history.reshape(-1).copy(), reward, done, info
