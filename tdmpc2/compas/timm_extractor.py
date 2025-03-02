import math
from typing import Any, Dict, List, Optional, Union

import timm
import torch
import torchvision
from torch import nn


def patch_timm_for_fx_tracing():
    """Patch timm to allow torch.fx tracing."""

    def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ):
        """From timm.layers.pos_embed.resample_abs_pose_embed.

        To avoid control flow using dynamic variables, the check returning early for same size
        is not executed.
        """
        # sort out sizes, assume square if old size not provided
        num_pos_tokens = posemb.shape[1]

        # REMOVED because this relies on dynamic variables:
        # num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        # if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        #    return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:, :
                                           num_prefix_tokens], posemb[:,
                                                                      num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1],
                                -1).permute(0, 3, 1, 2)
        posemb = nn.functional.interpolate(posemb,
                                           size=new_size,
                                           mode=interpolation,
                                           antialias=antialias)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=1)

        return posemb

    # Monkey patch method in vision transformer
    timm.models.vision_transformer.resample_abs_pos_embed = resample_abs_pos_embed


torch.fx.wrap("int")  # Needed to allow tracing with int()

patch_timm_for_fx_tracing()


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{
            f"resnet_block{i}": f"layer{i}"
            for i in range(1, 5)
        },
        **{
            f"vit_block{i + 1}": f"blocks.{i}"
            for i in range(12)
        },
        **{
            f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv"
            for i in range(12)
        },
        **{
            f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv"
            for i in range(12)
        },
        **{
            f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv"
            for i in range(12)
        },
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{
            f"layer{i}": f"resnet_block{i}"
            for i in range(1, 5)
        },
        **{
            f"blocks.{i}": f"vit_block{i + 1}"
            for i in range(12)
        },
        **{
            f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}"
            for i in range(12)
        },
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained,
                                            checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(
                model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(
                model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(model_name,
                                      pretrained=pretrained,
                                      checkpoint_path=checkpoint_path,
                                      **model_kwargs)
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(model_name,
                                                pretrained,
                                                checkpoint_path,
                                                model_kwargs,
                                                trials=trials + 1)

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {
                    k: v[:, 1:]
                    for k, v in outputs.items()
                }  # Remove CLS token
            outputs = {
                self.FEATURE_MAPPING[key]: value
                for key, value in outputs.items()
            }
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values"
                                                               in name):
                    feature_name = name.replace("queries", "keys").replace(
                        "values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3,
                        C // 3)  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs
