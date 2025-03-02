import torch
from torch import nn

from compas.timm_extractor import TimmExtractor

AVAILABLE_PATCH_SIZES = {
    1: {8, 14},
    2: {14},
}

AVAILABLE_MODEL_SIZES = {
    1: {'small', 'base'},
    2: {'small', 'base', 'large'},
}

MODEL_TEMPLATES = {
    1: 'vit_{}_patch{}_224_dino',
    2: 'vit_{}_patch{}_dinov2',
}

PATCHES_TO_NUM_PATCHES = {
    16: 196,
    8: 784,
    14: 256,
}

SIZE_TO_FEAT_DIM = {
    'small': 384,
    'base': 768,
    'large': 1024,
}

FEAT_TYPES = {
    'pre_norm': 'vit_block12',
    'atten': 'vit_block_keys12',
    'last_layer': 'vit_output'
}


class DINOEncoder(nn.Module):

    def __init__(self,
                 resolution: int,
                 patch_size: int,
                 model_size: str,
                 version: int = 1,
                 features_type= 'pre_norm',
                 frozen: bool = True):
        super().__init__()
        assert version in {1, 2}, "Only versions 1 to 2 are supported"
        assert patch_size in AVAILABLE_PATCH_SIZES[
            version], f"For version {version}, only {AVAILABLE_PATCH_SIZES[version]} are supported"
        #
        # available_features_types = get_args(FeaturesType)
        # assert features_type in available_features_types, f"features_type must be one of {available_features_types}"

        self.model_name = MODEL_TEMPLATES[version].format(
            model_size, patch_size)

        if isinstance(features_type, list) or isinstance(features_type, tuple):
            features = [FEAT_TYPES[feat_type] for feat_type in features_type]
        else:
            features = FEAT_TYPES[features_type]

        self.frozen = frozen
        self.model: nn.Module = TimmExtractor(
            model=self.model_name,
            pretrained=True,
            frozen=frozen,
            features=features,
            model_kwargs=dict(dynamic_img_size=True))
        # if checkpoint is not None:
        #     self.model.load_state_dict(torch.load(checkpoint)['teacher'])

        self.visual_res = resolution // patch_size

        if frozen:
            self.freeze()
        # pool_dim = (resolution // patch_size)**2
        # self.pooler = nn.Linear(pool_dim+1,pool_dim)

    def forward(self, image: torch.Tensor):
        # B = image.size(0)
        features = self.model(image)
        return features

    def freeze(self):
        self.model.eval()
        self.model.frozen = True
        self.model.requires_grad_(False)

    def unfreeze(self):
        self.model.frozen = False
        self.model.requires_grad_(True)

    def train(self, mode: bool = True):
        if self.frozen:
            return super().train(False)
        return super().train(mode)
