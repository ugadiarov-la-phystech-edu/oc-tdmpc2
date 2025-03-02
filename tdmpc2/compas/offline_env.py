from pydantic_conf.base_config import BaseConfig


class DefaultObsTransformsConfig(BaseConfig):
    resolution: int
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class VFlibObsTransformsConfig(DefaultObsTransformsConfig):
    pass