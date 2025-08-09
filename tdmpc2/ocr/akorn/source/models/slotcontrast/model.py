import torch
import torch.nn as nn

from ..slot_attention.resizer import Resizer, SoftToHardMask


class SlotContrastAkornSAur(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_output_transform: nn.Module, initializer: nn.Module,
                 decoder: nn.Module, latent_processor: nn.Module, is_encoder_frozen: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_output_transform = encoder_output_transform
        self.initializer = initializer
        self.decoder = decoder
        self.latent_processor = latent_processor
        self.is_encoder_frozen = is_encoder_frozen
        self.encoder = self.encoder.train(not self.is_encoder_frozen)
        self.encoder.requires_grad_(not self.is_encoder_frozen)
        self.resizer = Resizer(video_inputs=True, patch_inputs=True)
        self.soft_to_hard_mask = SoftToHardMask()
        self.num_slots = self.initializer.num_slots
        self.slot_dim = self.initializer.slot_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.is_encoder_frozen:
            self.encoder.train(False)

        return self

    def _get_features(self, images: torch.Tensor,):
        # backbone_features.shape -> batch_size, encoder_dim, n_patches_h, n_patches_w
        backbone_features = self.encoder(images, return_activation=True)

        # backbone_features.shape -> batch_size, n_patches_h x n_patches_w, encoder_dim
        backbone_features = backbone_features.flatten(start_dim=2).movedim(2, 1)

        # features.shape -> batch_size, n_patches_h x n_patches_w, projection_dim
        features = self.encoder_output_transform(backbone_features)

        return backbone_features, features

    def process_masks(
        self,
        masks: torch.Tensor,
        images: torch.Tensor,
    ):
        if masks is None:
            return None, None,

        masks_resized = self.resizer(masks, images)
        masks_resized_hard = self.soft_to_hard_mask(masks_resized)

        return masks_resized, masks_resized_hard

    def forward(self, images: torch.Tensor, actions: torch.Tensor, prior_slots=None, step_offset=0, reconstruct=False, masks=False, **kwargs):
        """
        Args:
            images (torch.Tensor): Image sequence of shape (B, sequence_length, C, H, W).
            actions (torch.Tensor): Action sequence of shape (B, sequence_length - 1, action_dim).

        Returns:
            torch.Tensor: Slots encoded at every time step of shape (B, sequence_length, num_slots, slot_dim)
            torch.Tensor: Reconstructed video frames by decoding and combining the slots of shape (B, sequence_length, C, H, W)
            torch.Tensor: Rendered objects of individual slots. Shape is (B, sequence_length, num_slots, C, H, W)
            torch.Tensor: Rendered object masks of individual slots. Shape is (B, sequence_length, num_slots, 1, H, W)
        """
        batch_size, sequence_length = images.shape[:2]
        slots_sequence = []
        slot_attention_masks_sequence = []

        # Initialize slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=batch_size) if prior_slots is None else self.latent_processor.predictor(prior_slots)
        backbone_features, features = self._get_features(images.flatten(end_dim=1))
        features = features.unflatten(dim=0, sizes=(batch_size, sequence_length))

        for t in range(sequence_length):
            slot_attention_output = self.latent_processor(state=predicted_slots, inputs=features[:, t], time_step=t + step_offset)
            predicted_slots = slot_attention_output['state_predicted']
            slots_sequence.append(slot_attention_output['state'])
            slot_attention_masks_sequence.append(slot_attention_output['corrector']['masks'])

        slots_sequence = torch.stack(slots_sequence, dim=1)
        result = {'slots_sequence': slots_sequence}
        if reconstruct or masks:
            decoder_output = self.decoder(slots_sequence.flatten(end_dim=1))

            if reconstruct:
                backbone_features = backbone_features.unflatten(dim=0, sizes=(batch_size, sequence_length))
                result['features_sequence'] = backbone_features
                result['features_reconstruction_sequence'] = decoder_output['reconstruction'].unflatten(dim=0, sizes=(batch_size, sequence_length))

            if masks:
                slot_attention_masks_sequence, slot_attention_masks_hard_sequence = self.process_masks(
                    torch.stack(slot_attention_masks_sequence, dim=1), images, )
                decoder_masks_sequence, decoder_masks_hard_sequence = self.process_masks(decoder_output['masks'].unflatten(dim=0, sizes=(batch_size, sequence_length)),
                                                                                         images,)
                result['slot_attention_masks_sequence'] = slot_attention_masks_sequence
                result['slot_attention_masks_hard_sequence'] = slot_attention_masks_hard_sequence
                result['decoder_masks_sequence'] = decoder_masks_sequence
                result['decoder_masks_hard_sequence'] = decoder_masks_hard_sequence

        return result
