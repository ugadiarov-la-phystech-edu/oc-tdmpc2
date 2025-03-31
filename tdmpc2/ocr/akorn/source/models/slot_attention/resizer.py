import math
from typing import Optional, Tuple

import torch
from torch import nn


class Resizer:
    """Module that takes image-based tensor and resizes it to an appropriate size.

    Args:
        size: Tuple of (height, width) to resize to. If unspecified, assume an additional
            input used to infer the size. The last two dimensions of this input are taken
            as height and width.
        patch_inputs: If true, assumes tensor to resize has format `(batch, [frames],
            channels, n_points)` instead of separate height, width dimensions.
        patch_outputs: If true, flatten spatial dimensions after resizing.
        video_inputs: If true, assume inputs have an additional video dimension
        channels_last: If true, assume channel dimension comes after spatial dimensions. Output will
            be in same format as input.
        resize_mode: Mode to use for resizing. For nearest neighbor resizing, specify
            nearest-exact instead of nearest.
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        patch_inputs: bool = False,
        patch_outputs: bool = False,
        video_inputs: bool = False,
        channels_last: bool = False,
        resize_mode: str = "bilinear",
    ):
        if resize_mode not in ("linear", "bilinear", "bicubic", "nearest-exact"):
            if resize_mode == "nearest":
                raise ValueError("Use resize mode `nearest-exact` instead of `nearest`")
            else:
                raise ValueError(f"Unsupported resize mode {resize_mode}")

        self.size = size
        self.patch_inputs = patch_inputs
        self.patch_outputs = patch_outputs
        self.video_inputs = video_inputs
        self.channels_last = channels_last
        self.n_expected_dims = 4 + (1 if video_inputs else 0) - (1 if patch_inputs else 0)
        self.resize_mode = resize_mode

    def __call__(
        self, inputs: torch.Tensor, size_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs.ndim != self.n_expected_dims:
            raise ValueError(
                f"Mask has {inputs.ndim} dimensions, but expected it to "
                f"have {self.n_expected_dims} dimensions."
            )

        if self.size is None:
            if size_tensor is None:
                raise ValueError("If size is unspecified, need to pass a tensor to take size from")
            size = size_tensor.shape[-2:]
        else:
            size = list(self.size)

        if self.video_inputs:
            batch, n_frames = inputs.shape[:2]
            inputs = inputs.flatten(0, 1)

        if self.channels_last:
            if self.patch_inputs:
                inputs = inputs.transpose(-1, -2)
            else:
                inputs = inputs.transpose(-1, -2).transpose(-2, -3)

        if self.patch_inputs:
            n_patches = inputs.shape[-1]
            ratio = size[1] / size[0]
            height = int(math.sqrt(n_patches / ratio))
            width = int(math.sqrt(n_patches * ratio))
            if height * width != n_patches:
                if height == width:
                    raise ValueError(
                        f"Can not reshape {n_patches} patches to square aspect ratio as it's not a "
                        "perfect square."
                    )
                raise ValueError(f"Can not reshape {n_patches} patches to aspect ratio {ratio}.")

            inputs = inputs.unflatten(-1, (height, width))

        dtype = inputs.dtype
        if inputs.dtype == torch.bool:
            inputs = inputs.to(torch.uint8)

        outputs = torch.nn.functional.interpolate(inputs, size=size, mode=self.resize_mode)

        if inputs.dtype != dtype:
            inputs = inputs.to(dtype)

        if self.resize_mode == "bicubic":
            outputs.clamp_(0.0, 1.0)  # Bicubic interpolation can get out of range

        if self.patch_outputs:
            outputs = outputs.flatten(-2, -1)

        if self.channels_last:
            if self.patch_outputs:
                outputs = outputs.transpose(-2, -1)
            else:
                outputs = outputs.transpose(-3, -2).transpose(-2, -1)

        if self.video_inputs:
            outputs = outputs.unflatten(0, (batch, n_frames))

        return outputs


class SoftToHardMask:
    """Module that converts masks from soft to hard."""

    def __init__(
        self, convert_one_hot: bool = True, use_threshold: bool = False, threshold: float = 0.5
    ):
        self.convert_one_hot = convert_one_hot
        self.use_threshold = use_threshold
        self.threshold = threshold

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        return soft_to_hard_mask(masks, self.convert_one_hot, self.use_threshold, self.threshold)


def soft_to_hard_mask(
    masks: torch.Tensor,
    convert_one_hot: bool = True,
    use_threshold: bool = False,
    threshold: float = 0.5,
):
    """Convert soft to hard masks."""
    # masks: batch [x n_frames] x n_channels x height x width
    assert masks.ndim == 4 or masks.ndim == 5
    min = torch.min(masks)
    max = torch.max(masks)
    if min < 0:
        raise ValueError(f"Minimum mask value should be >=0, but found {min.cpu().numpy()}")
    if max > 1:
        raise ValueError(f"Maximum mask value should be <=1, but found {max.cpu().numpy()}")

    if use_threshold:
        masks = masks > threshold

    if convert_one_hot:
        mask_argmax = torch.argmax(masks, dim=-3)
        masks = nn.functional.one_hot(mask_argmax, masks.shape[-3]).to(torch.float32)
        masks = masks.transpose(-1, -2).transpose(-2, -3)  # B, [F,] H, W, C -> B, [F], C, H, W

    return masks
