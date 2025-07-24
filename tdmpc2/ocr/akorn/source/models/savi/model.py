from ..savi import Predictor
import torch
import torch.nn as nn

from ..slot_attention.resizer import Resizer, SoftToHardMask


class AkornSAVi(nn.Module):
    def __init__(self, encoder: nn.Module, features_projector: nn.Module, initializer: nn.Module, slot_attention: nn.Module,
                 decoder: nn.Module, predictor: Predictor, is_encoder_frozen: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.features_projector = features_projector
        self.initializer = initializer
        self.slot_attention = slot_attention
        self.decoder = decoder
        self.predictor = predictor
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

        # features.shape -> batch_size, encoder_dim, n_patches_h, n_patches_w
        features = self.encoder(images, return_activation=True)

        # x.shape -> batch_size, n_patches_h x n_patches_w, encoder_dim
        x = features.flatten(start_dim=2).movedim(2, 1)

        # x.shape -> batch_size, n_patches_h x n_patches_w, projection_dim
        slot_attention_input = self.features_projector(x)

        return features, slot_attention_input

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
        slots_sequence = []
        features_sequence = []
        features_reconstruction_sequence = []
        slot_attention_masks_sequence = []
        decoder_masks_sequence = []

        sequence_length = images.shape[1]

        # Initialize slots by randomly sampling them or encoding some representations (e.g. BBox)
        predicted_slots = self.initializer(batch_size=images.shape[0]) if prior_slots is None else self.predictor(prior_slots, actions[:, 0])

        # Recursively map video frames into slots.
        for t in range(sequence_length):
            imgs = images[:, t]
            img_feats, sloat_attention_input = self._get_features(imgs)
            slot_attention_output = self.slot_attention(slots=predicted_slots, image_features=sloat_attention_input, step=t + step_offset)
            # slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None, step: Optional[int] = 0)
            slots = slot_attention_output['slots']
            if t < sequence_length - 1:
                predicted_slots = self.predictor(slots, actions[:, t])

            slots_sequence.append(slots)
            if reconstruct or masks:
                decoder_output = self.decoder(slots)
                if reconstruct:
                    features_sequence.append(img_feats)
                    features_reconstruction_sequence.append(decoder_output['reconstruction'].movedim(2, 1).reshape_as(img_feats))
                if masks:
                    slot_attention_masks_sequence.append(slot_attention_output['masks'])
                    decoder_masks_sequence.append(decoder_output['masks'])


        result = {'slots_sequence': torch.stack(slots_sequence, dim=1)}
        if reconstruct:
            result['features_sequence'] = torch.stack(features_sequence, dim=1)
            result['features_reconstruction_sequence'] = torch.stack(features_reconstruction_sequence, dim=1)

        if masks:
            slot_attention_masks_sequence = torch.stack(slot_attention_masks_sequence, dim=1)
            slot_attention_masks_sequence, slot_attention_masks_hard_sequence = self.process_masks(slot_attention_masks_sequence, images, )
            decoder_masks_sequence = torch.stack(decoder_masks_sequence, dim=1)
            decoder_masks_sequence, decoder_masks_hard_sequence = self.process_masks(decoder_masks_sequence, images, )
            result['slot_attention_masks_sequence'] = slot_attention_masks_sequence
            result['slot_attention_masks_hard_sequence'] = slot_attention_masks_hard_sequence
            result['decoder_masks_sequence'] = decoder_masks_sequence
            result['decoder_masks_hard_sequence'] = decoder_masks_hard_sequence

        return result
