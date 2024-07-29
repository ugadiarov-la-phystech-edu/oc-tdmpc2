import math
import os
from functools import partial

# Types
from typing import TypeVar, Optional

import torch
import torchvision.transforms
import wandb
import omegaconf
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ocr.dinosaur.decoding import PatchDecoder
from ocr.dinosaur.neural_networks import build_two_layer_mlp, build_mlp
from ocr.dinosaur.neural_networks.positional_embedding import DummyPositionEmbed
from ocr.dinosaur.neural_networks.wrappers import Sequential
from ocr.dinosaur.conditioning import RandomConditioning
from ocr.dinosaur.feature_extractors.timm import TimmFeatureExtractor
from ocr.dinosaur.perceptual_grouping import SlotAttentionGrouping

Tensor = TypeVar("torch.tensor")
NN = TypeVar("torch.nn")

# [B, D, H, W] -> [B, N, D]
img_to_slot = lambda x: x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


# [B, N, D] -> [B, D, H, W]
def slot_to_img(slot):
    B, N, D = slot.shape
    size = int(math.sqrt(N))
    return slot.reshape(B, size, size, D).permute(0, 3, 1, 2)


def get_ocr_checkpoint_path(config):
    if config.local_file:
        return Path(__file__).resolve().parents[1] / config.local_file

    entity = config.entity
    project = config.project
    run_id = config.run_id
    file_path = config.file
    ocr_dir = Path(wandb.run.dir) / "ocr_checkpoints"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run.file(file_path).download(root=ocr_dir, replace=True)
    return ocr_dir / file_path


def get_log_prefix(config):
    prefix = ""
    if config.ocr.name == "VAE":
        if config.ocr.use_cnn_feat:
            prefix = f"{config.ocr.name}N{config.ocr.cnn_feat_size ** 2}"
        else:
            prefix = f"{config.ocr.name}"
    elif config.ocr.name == "SlotAttn" or config.ocr.name == "SLATE":
        prefix = f"{config.ocr.name}N{config.ocr.slotattr.num_slots}"
    elif config.ocr.name == "MoNet":
        prefix = f"{config.ocr.name}N{config.ocr.num_slots}"
    else:
        prefix = f"{config.ocr.name}"
    if hasattr(config, "pooling"):
        if config.pooling.ocr_checkpoint.run_id != "":
            prefix = "Pretrained-" + prefix
        if config.pooling.learn_aux_loss:
            prefix += f"Aux"
        if config.pooling.learn_downstream_loss:
            prefix += f"FineTune"
        prefix += f"-{config.pooling.name}"
    return prefix


def init_wandb(config, log_name, tags="", sync_tensorboard=None, monitor_gym=None):
    if config.wandb.offline:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_MODE"] = "online"
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=wandb.config,
        name=log_name,
        dir=str(Path.cwd().parent.parent),
        save_code=True,
        sync_tensorboard=sync_tensorboard,
        monitor_gym=monitor_gym,
        resume="allow",
        id=config.wandb.run_id,
        tags=tags,
    )
    model_dir = Path(run.dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)


# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Taken from https://github.com/lcswillems/torch-ac/blob/master/torch_ac/utils/dictlist.py
class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


def preprocessing_obs(obs, device, type="image"):
    ret = torch.Tensor(obs.copy()).to(device).unsqueeze(0)
    if type == "image":
        return ret.permute(0, 3, 1, 2) / 255
    elif type == "state":
        return ret


# upload batch to working device
def to_device(batch, device):
    if type(batch) == type([]):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
    elif type(batch) == type({}):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
    else:
        batch = batch.to(device)
    return batch


# get_item from pytorch tensor
def get_item(x):
    if len(x.shape) == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()


# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0, dtype=np.uint8
)


# Taken from https://github.com/singhgautam/slate/blob/master/slate.py
def visualize(images):
    B, _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        if len(_img.shape) == 4:
            viz_imgs.append(_img)
        else:
            viz_imgs += [object_image.expand_as(viz_imgs[0]) for object_image in torch.unbind(_img, dim=1)]
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs


# Load model and params
def load(model, agent_training=False, resume_checkpoint=None, resume_run_path=None):
    checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location=next(model._module.parameters()).device
        )
    elif resume_run_path is not None:
        checkpoint = torch.load(
            wandb.restore(
                "checkpoints/model_latest.pth", run_path=resume_run_path
            ).name,
            map_location=next(model._module.parameters()).device,
        )
    else:
        model_checkpoint = Path(wandb.run.dir) / "checkpoints" / "model_latest.pth"
        if model_checkpoint.exists():
            checkpoint = torch.load(
                model_checkpoint, map_location=next(model._module.parameters()).device
            )

    if checkpoint is not None:
        step = checkpoint["step"]
        if agent_training:
            episode = checkpoint["episode"]
        else:
            epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
        model.load(checkpoint)

    else:
        step = 0
        if agent_training:
            episode = 0
        else:
            epoch = 0
            best_val_loss = 1e10

    if agent_training:
        return step, episode
    else:
        return step, epoch, best_val_loss


# Save model and params
def save(
        model,
        step=0,
        epoch=0,
        best_val_loss=1e5,
        episode=0,
        agent_training=False,
        best=False,
):
    sub_dir = "checkpoints"
    model_dir = Path(wandb.run.dir) / sub_dir
    if agent_training:
        checkpoint = {"step": step, "episode": episode}
    else:
        checkpoint = {"step": step, "epoch": epoch, "best_val_loss": best_val_loss}
    checkpoint.update(model.save())
    torch.save(checkpoint, model_dir / f"model_{step}.pth")
    wandb.save(f"{sub_dir}/model_{step}.pth")
    torch.save(checkpoint, model_dir / f"model_latest.pth")
    wandb.save(f"{sub_dir}/model_latest.pth")
    if best:
        torch.save(checkpoint, model_dir / f"model_best.pth")
        wandb.save(f"{sub_dir}/model_best.pth")


# hungarian matching
def hungarian_matching(target, input, return_diff_mat=False):
    tN, tD = target.shape
    iN, iD = input.shape
    assert tN == iN and tD == iD
    diff_mat = np.zeros((tN, iN))
    for t in range(tN):
        for i in range(iN):
            diff_mat[t, i] = torch.norm(target[t] - input[i], p=1).item()
    _, col_ind = linear_sum_assignment(diff_mat)
    if return_diff_mat:
        return torch.LongTensor(col_ind).to(target.device), diff_mat[:, col_ind]
    else:
        return torch.LongTensor(col_ind).to(target.device)


# calculate ARI
def calculate_ari(true_masks, pred_masks):
    true_masks = true_masks.flatten(2)
    pred_masks = pred_masks.flatten(2)

    true_mask_ids = get_item(torch.argmax(true_masks, dim=1))
    pred_mask_ids = get_item(torch.argmax(pred_masks, dim=1))

    aris = []
    for b in range(true_mask_ids.shape[0]):
        aris.append(adjusted_rand_score(true_mask_ids[b], pred_mask_ids[b]))

    return aris


# change img numpy array to torch Tensor
def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)


class SlotExtractor:
    def __init__(self, model, device):
        self._model = model
        self._device = device
        self._model.to(device)

    def get_slots_dim(self):
        return self._model.get_slots_dim()

    def __call__(self, images, prev_slots, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if prev_slots is not None and len(prev_slots.shape) == 2:
            batch_prev_slots = prev_slots[np.newaxis, ...]
        else:
            batch_prev_slots = prev_slots

        batch_images = obs_to_tensor(batch_images, self._device)
        if batch_prev_slots is not None:
            batch_prev_slots = obs_to_tensor(batch_prev_slots, self._device)

        slots = self._model(batch_images, prev_slots=batch_prev_slots).detach()
        if len(images.shape) == 3:
            slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots

    def visualize(self, images, slots, with_attns=False, normalize_slots=False, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if len(slots.shape) == 2:
            batch_slots = slots[np.newaxis, ...]
        else:
            batch_slots = slots

        batch_images = obs_to_tensor(batch_images, self._device)
        if batch_slots is not None:
            batch_slots = obs_to_tensor(batch_slots, self._device)

        attns = self._model.visualize(batch_images, batch_slots, with_attns=with_attns,
                                      normalize_slots=normalize_slots).detach()

        if to_numpy:
            attns = attns.cpu().numpy()

        return attns


class Dinosaur(torch.nn.Module):
    def __init__(self, dino_model_name, n_slots, slot_dim, intput_feature_dim, num_patches, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dino_model_name = dino_model_name
        self._n_slots = n_slots
        self._slot_dim = slot_dim
        self._input_feature_dim = intput_feature_dim
        self._num_patches = num_patches
        self._features = features
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.feature_extractor = TimmFeatureExtractor(model_name=self._dino_model_name, feature_level=12,
                                                      pretrained=True, freeze=True)
        self.conditioning = RandomConditioning(object_dim=self._slot_dim, n_slots=self._n_slots, learn_mean=True,
                                               learn_std=True)

        pos_embedding = Sequential(DummyPositionEmbed(),
                                   build_two_layer_mlp(input_dim=self._input_feature_dim, output_dim=self._slot_dim,
                                                       hidden_dim=self._input_feature_dim, initial_layer_norm=True))
        ff_mlp = build_two_layer_mlp(input_dim=self._slot_dim, output_dim=self._slot_dim, hidden_dim=4 * self._slot_dim,
                                     initial_layer_norm=True, residual=True)
        self.perceptual_grouping = SlotAttentionGrouping(feature_dim=self._slot_dim, object_dim=self._slot_dim,
                                                         ff_mlp=ff_mlp,
                                                         positional_embedding=pos_embedding, use_projection_bias=False,
                                                         use_implicit_differentiation=False,
                                                         use_empty_slot_for_masked_slots=False)

        decoder = partial(build_mlp, features=self._features)
        self.object_decoder = PatchDecoder(object_dim=self._slot_dim, output_dim=self._input_feature_dim,
                                           num_patches=self._num_patches, decoder=decoder, )

    def get_slots_dim(self):
        return self._n_slots, self._slot_dim

    def forward(self, image, prev_slots=None):
        image = self._normalization(image)
        feature_extraction_output = self.feature_extractor(image)
        conditioning_output = prev_slots
        if conditioning_output is None:
            conditioning_output = self.conditioning(feature_extraction_output.features.size()[0])

        perceptual_grouping_output = self.perceptual_grouping(feature_extraction_output, conditioning_output)
        # patch_reconstruction_output = self._patch_decoder(perceptual_grouping_output.objects,
        #                                                   feature_extraction_output.features, image)

        return perceptual_grouping_output.objects


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
        pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1:]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)


class ARIMetric:
    """Computes ARI metric."""

    def __init__(
            self,
            foreground: bool = True,
            convert_target_one_hot: bool = False,
            ignore_overlaps: bool = False,
            background_dim: int = 0
    ):
        super().__init__()
        self.foreground = foreground
        self.background_dim = background_dim
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.values = 0
        self.total = 0

    def update(
            self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target, bg_dim=self.background_dim)
        else:
            ari = adjusted_rand_index(prediction, target)

        self.values += ari.sum().item()
        self.total += len(ari)

    def compute(self):
        return self.values / self.total
