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


class DynamicDDLPExtractorWrapper(gym.Wrapper):
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
        dlp_output = self.ddlp(x, deterministic=True, x_prior=x, warmup=False, noisy=False, forward_dyn=False,
                               train_enc_prior=self.train_enc_prior, num_static_frames=self.num_static_frames)

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


class StaticDDLPExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses DDLP model.
    """

    def __init__(self, env, model, device, num_static_frames, train_enc_prior, num_frames):
        super().__init__(env)
        assert env.observation_space.shape[:-1] == (model.image_size,
                                                    model.image_size), f'Expected image size: {model.image_size}. Actual image shape: {env.observation_space.shape}'

        self.ddlp = model
        self.device = device
        self.num_frames = num_frames
        self.num_static_frames = num_static_frames
        self.train_enc_prior = train_enc_prior
        n_particles = model.n_kp_enc
        self.observation_space = gym.spaces.Dict({
            'fg': gym.spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.num_frames, n_particles, self.ddlp.get_dlp_features_dim()),
                                 dtype=np.float32),
            'bg': gym.spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.num_frames, self.ddlp.bg_learned_feature_dim,),
                                 dtype=np.float32),
        })
        self.fg = np.zeros((self.num_frames, n_particles, self.ddlp.get_dlp_features_dim()), dtype=np.float32)
        self.bg = np.zeros((self.num_frames, self.ddlp.bg_learned_feature_dim), dtype=np.float32)
        self.z_prev = None
        self.z_scale_prev = None
        self.cropped_objects_prev = None

    def _encode_on_episode_start(self, frame):
        x = torch.as_tensor(frame, dtype=torch.float32, device=self.device) / 255.

        x = x.permute(2, 0, 1).unsqueeze(0).expand(self.ddlp.timestep_horizon + 1, -1, -1, -1).unsqueeze(0)
        dlp_output = self.ddlp(x, deterministic=True, x_prior=x, warmup=False, noisy=False, predict_next=False,
                               sequential=True, train_enc_prior=self.train_enc_prior, num_static_frames=self.num_static_frames)

        self.z_prev = dlp_output['z'][-1:]
        self.z_scale_prev = dlp_output['z_scale'][-1:]
        self.cropped_objects_prev = dlp_output['cropped_objects_original'][-1:]

        fg = self.ddlp.get_dlp_rep(dlp_output['z'][-1], dlp_output['mu_scale'][-1], dlp_output['mu_depth'][-1],
                                   dlp_output['mu_features'][-1], dlp_output['obj_on'][-1].unsqueeze(dim=-1))

        return {'fg': fg.detach().cpu().numpy(), 'bg': dlp_output['z_bg'][-1].detach().cpu().numpy()}

    def _encode_on_step(self, new_frame):
        x = torch.as_tensor(new_frame, dtype=torch.float32, device=self.device) / 255.
        x = x.permute(2, 0, 1).unsqueeze(0)
        fg_dict = self.ddlp.fg_module.encode_all(x, deterministic=False, warmup=False, noisy=False, kp_init=self.z_prev,
                                                 cropped_objects_prev=self.cropped_objects_prev.flatten(end_dim=1),
                                                 scale_prev=self.z_scale_prev, refinement_iter=False)

        z = fg_dict['z']
        z_scale = fg_dict['z_scale']
        cropped_objects = fg_dict['cropped_objects']
        z_obj_on = fg_dict['obj_on']
        mu_scale = fg_dict['mu_scale']
        mu_depth = fg_dict['mu_depth']
        mu_features = fg_dict['mu_features']
        bg_enc_mask = self.ddlp.get_bg_mask_from_particle_glimpses(z, z_obj_on, mask_size=x.shape[-1])
        bg_dict = self.ddlp.bg_module(x, bg_enc_mask, deterministic=False)
        z_bg = bg_dict['z_bg']

        self.z_prev = z
        self.z_scale_prev = z_scale
        self.cropped_objects_prev = cropped_objects

        return {'fg': self.ddlp.get_dlp_rep(z, mu_scale, mu_depth, mu_features,
                                            z_obj_on.unsqueeze(dim=-1)).detach().cpu().numpy(),
                'bg': z_bg.detach().cpu().numpy()}

    def _get_observation(self):
        return {'fg': self.fg.copy(), 'bg': self.bg.copy()}


    def reset(self):
        frame = self.env.reset()
        representation = self._encode_on_episode_start(frame)
        self.fg[:] = representation['fg']
        self.bg[:] = representation['bg']

        return self._get_observation()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        representation = self._encode_on_step(frame)
        self.fg[:-1] = self.fg[1:]
        self.fg[-1] = representation['fg']
        self.bg[:-1] = self.bg[1:]
        self.bg[-1] = representation['bg']

        return self._get_observation(), reward, done, info
