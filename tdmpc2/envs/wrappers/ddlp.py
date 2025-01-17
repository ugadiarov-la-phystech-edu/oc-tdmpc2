import gym
import numpy as np
import torch


def get_dlp_rep(dlp_output):
    pixel_xy = dlp_output['z']
    scale_xy = dlp_output['mu_scale']
    depth = dlp_output['mu_depth']
    visual_features = dlp_output['mu_features']
    transp = dlp_output['obj_on'].unsqueeze(dim=-1)
    rep = torch.cat((pixel_xy, scale_xy, depth, visual_features, transp,), dim=-1)
    return rep


class DDLPExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses DDLP model.
    """

    def __init__(self, env, model, device, num_static_frames, train_enc_prior):
        super().__init__(env)
        assert env.observation_space.shape[:-1] == (model.image_size, model.image_size), f'Expected image size: {model.image_size}. Actual image shape: {env.observation_space.shape}'

        self.ddlp = model
        self.device = device
        self.num_static_frames = num_static_frames
        self.train_enc_prior = train_enc_prior
        n_particles = model.n_kp_enc
        timestep_horizon = self.ddlp.timestep_horizon
        self.observation_space = gym.spaces.Dict({
            'fg': gym.spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(timestep_horizon, n_particles, self.ddlp.get_dlp_features_dim()),
                                         dtype=np.float32),
            'bg': gym.spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(timestep_horizon, self.ddlp.bg_learned_feature_dim),
                                         dtype=np.float32),
        })
        self.frames = np.zeros((self.ddlp.timestep_horizon, *env.observation_space.shape),
                               dtype=env.observation_space.dtype)
        self.actions = np.zeros((self.ddlp.timestep_horizon, *env.action_space.shape), dtype=env.action_space.dtype)

    def _encode(self):
        x = torch.as_tensor(self.frames, dtype=torch.float32, device=self.device) / 255.
        x = x.permute(0, 3, 1, 2).unsqueeze(0)
        dlp_output = self.ddlp(x, deterministic=True, x_prior=x, warmup=False, noisy=False, forward_dyn=True,
                               train_enc_prior=self.train_enc_prior, num_static_frames=self.num_static_frames,
                               predict_next=False)

        fg = self.ddlp.get_dlp_rep(dlp_output['z'], dlp_output['mu_scale'], dlp_output['mu_depth'],
                                   dlp_output['mu_features'], dlp_output['obj_on'].unsqueeze(dim=-1))

        return {'fg': fg.detach().cpu().numpy(),
                'bg': dlp_output['z_bg'].detach().cpu().numpy()}

    def get_actions(self):
        return self.actions.copy()

    def reset(self):
        frame = self.env.reset()
        self.frames[:] = frame
        self.actions[:] = 0
        return self._encode()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        self.actions[:-1] = self.actions[1:]
        self.actions[-1] = action
        return self._encode(), reward, done, info
