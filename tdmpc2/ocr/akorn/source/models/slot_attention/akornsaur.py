from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn
from torch.nn.modules.module import T

from ocr.akorn.source.models.slot_attention.resizer import Resizer, SoftToHardMask


class AkornSAur(nn.Module):
    def __init__(self, encoder: nn.Module, features_projector: nn.Module, initializer: nn.Module, slot_attention: nn.Module,
                 decoder: nn.Module, is_encoder_frozen: bool = True):
        super().__init__()
        self.encoder = encoder
        self.features_projector = features_projector
        self.initializer = initializer
        self.slot_attention = slot_attention
        self.decoder = decoder
        self.is_encoder_frozen = is_encoder_frozen
        self.encoder = self.encoder.train(not self.is_encoder_frozen)
        self.encoder.requires_grad_(not self.is_encoder_frozen)
        self.resizer = Resizer(patch_inputs=True)
        self.soft_to_hard_mask = SoftToHardMask()

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        if self.is_encoder_frozen:
            self.encoder.train(False)

        return self

    def get_slots_dim(self):
        return self.initializer.n_slots, self.initializer.dim

    def _get_features_slot_attention(self, images: torch.Tensor, slots_initial: torch.Tensor = None):
        # images.shape -> batch_size, n_channels, height, width
        batch_size = images.size()[0]

        # features.shape -> batch_size, encoder_dim, n_patches_h, n_patches_w
        features = self.encoder(images, return_activation=True)

        # x.shape -> batch_size, n_patches_h x n_patches_w, encoder_dim
        x = features.flatten(start_dim=2).movedim(2, 1)

        # x.shape -> batch_size, n_patches_h x n_patches_w, projection_dim
        x = self.features_projector(x)

        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        return features, self.slot_attention(slots_initial, x)

    def forward(self, images: torch.Tensor, slots_initial: torch.Tensor = None):
        features, slot_attention_output = self._get_features_slot_attention(images, slots_initial)
        decoder_output = self.decoder(slot_attention_output['slots'])

        return {'features': features, 'slot_attention': slot_attention_output, 'decoder': decoder_output}

    def get_decoder_masks_by_slots(self, images: torch.Tensor, slots: torch.Tensor):
        return self.process_masks(self.decoder(slots)["masks"], images)

    def get_slots(self, images: torch.Tensor, slots_initial: torch.Tensor = None):
        _, slot_attention_output = self._get_features_slot_attention(images, slots_initial)
        return slot_attention_output['slots']

    def process_masks(
        self,
        masks: torch.Tensor,
        images: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],]:
        if masks is None:
            return None, None,

        masks_resized = self.resizer(masks, images)
        masks_resized_hard = self.soft_to_hard_mask(masks_resized)

        return masks_resized, masks_resized_hard

    def aux_forward(self, images: torch.Tensor, forward_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = forward_outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard = self.process_masks(decoder_masks, images,)

        slot_attention_masks = forward_outputs["slot_attention"].get("masks")
        slot_attention_masks, slot_attention_masks_hard = self.process_masks(slot_attention_masks, images,)

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_hard
        if slot_attention_masks is not None:
            aux_outputs["slot_attention_masks"] = slot_attention_masks
        if slot_attention_masks_hard is not None:
            aux_outputs["slot_attention_masks_hard"] = slot_attention_masks_hard

        return aux_outputs

    def step(self, images: torch.Tensor, do_predict_masks: bool = False):
        outputs = self.forward(images)
        aux_outputs = {}
        if do_predict_masks:
            aux_outputs = self.aux_forward(images, outputs)

        features = outputs['features']
        reconstruction = outputs['decoder']['reconstruction'].movedim(2, 1).reshape_as(features)
        loss = nn.functional.mse_loss(reconstruction, features)

        return loss, aux_outputs
