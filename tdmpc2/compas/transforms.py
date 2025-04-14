import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as vF

from compas.offline_env import VFlibObsTransformsConfig, DefaultObsTransformsConfig


class TorchNormalize:
    """Normalize the image with mean and std."""

    def __init__(self, mean=0.5, std=0.5, minmax=True):
        self.minmax = minmax
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean)[:, None, None]  # [1, 1, 3]
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std)[:, None, None]  # [1, 1, 3]
        self.mean = mean
        self.std = std

    def normalize_image(self, image: torch.Tensor):
        if self.minmax:
            image = image.float() / 255.
        image = (image - self.mean) / self.std
        return image

    def denormalize_image(self, image: torch.Tensor):
        # simple numbers
        if isinstance(self.mean, (int, float)) and \
                isinstance(self.std, (int, float)):
            image = image * self.std + self.mean
            return image.clamp(0, 1)
        # need to convert the shapes
        mean = self.mean.to(image.device).to(image.dtype)  # [3 1 1]

        std = self.std.to(image.device).to(image.dtype)  # [3 1 1]

        if len(image.shape) == 4:  # [B, C, H, W] or [B, H, W, C], batch dim
            mean = mean[None]
            std = std[None]
        elif len(image.shape) == 5:  # B T C H W
            mean = mean[None, None]
            std = std[None, None]
        image = image * std + mean

        return image.clamp(0, 1)

    def __call__(self, image: torch.Tensor):
        # [H, W, C]
        return self.normalize_image(image)


class Normalize:
    """Normalize the image with mean and std."""

    def __init__(self, mean=0.5, std=0.5):
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean, dtype=np.float32)[None, None]  # [1, 1, 3]
        if isinstance(std, (list, tuple)):
            std = np.array(std, dtype=np.float32)[None, None]  # [1, 1, 3]
        self.mean = mean
        self.std = std

    def normalize_image(self, image: np.ndarray):
        image = image.astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

    def denormalize_image(self, image: torch.Tensor):
        # simple numbers
        if isinstance(self.mean, (int, float)) and \
                isinstance(self.std, (int, float)):
            image = image * self.std + self.mean
            return image.clamp(0, 1)
        # need to convert the shapes
        mean = image.new_tensor(self.mean.squeeze())  # [3]
        std = image.new_tensor(self.std.squeeze())  # [3]
        if image.shape[-1] == 3:  # C last
            mean = mean[None, None]  # [1, 1, 3]
            std = std[None, None]  # [1, 1, 3]
        else:  # C first
            mean = mean[:, None, None]  # [3, 1, 1]
            std = std[:, None, None]  # [3, 1, 1]
        if len(image.shape) == 4:  # [B, C, H, W] or [B, H, W, C], batch dim
            mean = mean[None]
            std = std[None]
        elif len(image.shape) == 5:  # B T C H W
            mean = mean[None, None]
            std = std[None, None]
        image = image * std + mean
        return image.clamp(0, 1)

    def __call__(self, image: torch.Tensor):
        # [H, W, C]
        return self.normalize_image(image)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DefaultObsTransforms:

    def __init__(
            self,
            resolution: int,
            mean= IMAGENET_DEFAULT_MEAN,
            std= IMAGENET_DEFAULT_STD
    ) -> None:
        self.resolution = resolution
        self.mean = mean
        self.std = std
        self.normalize = Normalize(mean, std)
        self.mask_resize = transforms.Resize((self.resolution, self.resolution),
                                             interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        self.transforms = transforms.Compose([
            self.normalize,
            transforms.ToTensor(),
            transforms.Resize((self.resolution, self.resolution)),
        ])

    def transform_masks(self, masks: torch.Tensor):
        self.mask_resize(masks)

    def __call__(self, x: torch.Tensor):
        return self.transforms(x)


class VFlipObsTransforms(DefaultObsTransforms):

    def __call__(self, x: torch.Tensor):
        x = super().__call__(x)
        return vF.vflip(x)

    def transform_masks(self, masks: torch.Tensor):
        masks = super().__call__(masks)
        return vF.vflip(masks)


def get_obs_transform(obs_transforms):
    if isinstance(obs_transforms, VFlibObsTransformsConfig):
        return VFlipObsTransforms(**obs_transforms.shallow_dump())
    elif isinstance(obs_transforms, DefaultObsTransformsConfig):
        return DefaultObsTransforms(**obs_transforms.shallow_dump())
    else:
        raise ValueError(f"Unknown transforms: {obs_transforms.config_name}")


class PrintTransform:
    def __init__(self, message: str):
        self.message = message

    def __call__(self, x: torch.Tensor):
        print(self.message)
        print(x.shape)
        print(torch.max(x))
        print(torch.min(x))
        print('======================================================')
        return x
