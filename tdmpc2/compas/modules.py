from pathlib import Path
from typing_extensions import Self

import torch
from torch import nn

from compas.base_config import BaseConfig
from compas.dino import DINOEncoder
from compas.transformer_compas import ExtractorModel


def resolve_path(path_var):
    if isinstance(path_var, Path):
        return path_var
    elif isinstance(path_var, str):
        return Path(path_var)
    else:
        raise ValueError(f'Unexpected input type: {type(path_var)}')


class CompasSlotsExtractorAdapterConfig(BaseConfig):
    weights_path: object
    encoder_config: object
    num_slots: int
    slots_dim: int
    max_timestep: int
    feat_dim: int
    num_patches: int
    num_heads: int = 4
    num_layers: int = 3
    hidden_mult: int = 4


SIZE_TO_FEAT_DIM = {
    'small': 384,
    'base': 768,
    'large': 1024,
}

class DinoEncoderConfig(BaseConfig):
    resolution: int
    patch_size: int
    model_size: object
    version: int
    features_type: tuple = ('pre_norm',)
    frozen: bool = True

    def resolve_num_patches(self):
        return (self.resolution // self.patch_size) ** 2

    def resolve_feat_dim(self):
        return SIZE_TO_FEAT_DIM[self.model_size]


class CompasExtractorAdapter(nn.Module):
    def __init__(self,
                 weights_path,
                 num_slots: int,
                 encoder_config,
                 slots_dim: int,
                 max_timestep: int,
                 feat_dim: int,
                 num_patches: int,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 hidden_mult: int = 4, ):
        super().__init__()
        self.weights_path = weights_path
        self.num_slots = num_slots
        self.slots_dim = slots_dim
        self.max_timestep = max_timestep
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_mult = hidden_mult

        self.slots_pos_emb = nn.Parameter(torch.zeros(1, 1, num_slots, slots_dim),
                                          requires_grad=False)
        self.extractor_model = ExtractorModel(num_slots,
                                              slots_dim,
                                              max_timestep,
                                              feat_dim,
                                              num_patches,
                                              self.slots_pos_emb,
                                              num_heads,
                                              num_layers,
                                              hidden_mult)

        self.encoder = DINOEncoder(**encoder_config.model_dump()).eval()

        self.requires_grad_(False)
        self.eval()

        self._load_state_from_method(self.weights_path)

    # def collect(self, video: torch.Tensor):
    @torch.inference_mode()
    def collect(self, video: torch.Tensor):
        T = video.size(1)
        features = self.encoder(video.flatten(0, 1)).unflatten(0, (-1, T))
        slots, attns = self.extractor_model.forward_autoregressive(features)
        return slots

    @torch.inference_mode()
    def get_start_slots(self, framestack: torch.Tensor):
        features = self.encoder(framestack).unsqueeze(0)
        slots, attns = self.extractor_model.get_start_slots(features)
        return slots, attns

    @torch.inference_mode()
    def get_next_slot(self, next_obs: torch.Tensor, prev_slots: torch.Tensor):
        features = self.encoder(next_obs.unsqueeze(0))
        next_slots, _ = self.extractor_model.get_next_slot(features, prev_slots.unsqueeze(0))
        return next_slots



    def _load_state_from_method(self, weights_path):
        weights_path = resolve_path(weights_path)
        state = torch.load(weights_path)['state_dict']
        extractor_model_state = {k.replace('slots_transformer.extractor_model.', ''): v for k, v in state.items() if
                                 k.startswith('slots_transformer.extractor_model.')}

        encoder_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
        slots_pos_embed = [v for k, v in state.items() if k.startswith('slots_transformer.slots_pos_emb')][0]
        self.slots_pos_emb.data.copy_(slots_pos_embed)

        self.extractor_model.load_state_dict(extractor_model_state)
        self.encoder.load_state_dict(encoder_state)

    def train(self, mode: bool = True) -> Self:
        return super().train(False)
