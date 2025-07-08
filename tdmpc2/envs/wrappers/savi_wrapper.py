import numpy as np
import torch

from sold.modeling.savi.model import SAVi


def load_savi_module(savi: SAVi, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    savi.load_state_dict({k[len('savi.'):]: v for k, v in checkpoint['state_dict'].items()})

    return savi


class SlotExtractor:
    def __init__(self, model: SAVi, device):
        self._savi = model
        self._device = device
        self._savi.to(device)

    def get_slots_dim(self):
        return self._savi.num_slots, self._savi.slot_dim

    def __call__(self, images, prev_slots, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, np.newaxis, ...]
        elif len(images.shape) == 4:
            batch_images = images[np.newaxis, ...]
        else:
            raise ValueError(f'Unexpected batch images dimension: {images.shape}')

        if prev_slots is not None and len(prev_slots.shape) == 2:
            batch_prev_slots = prev_slots[np.newaxis, ...]
        else:
            batch_prev_slots = prev_slots

        batch_images = torch.Tensor(batch_images.transpose(0, 1, 4, 2, 3)).to(self._device) / 255.0
        if batch_prev_slots is not None:
            batch_prev_slots = torch.Tensor(batch_prev_slots).to(self._device)

        slots = self._savi(batch_images, actions=torch.empty((0, 1)), prior_slots=batch_prev_slots, step_offset=0 if prev_slots is None else 1, reconstruct=False).detach()
        # Expect len(images.shape) in [3, 4]
        if len(images.shape) == 3:
            slots = slots[0]

        slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots