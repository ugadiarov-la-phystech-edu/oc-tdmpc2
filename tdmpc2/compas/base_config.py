from itertools import chain
from typing import Any

from pydantic import BaseModel, ConfigDict

BASE_CONFIG = ConfigDict(validate_assignment=True)


class BaseConfig(BaseModel):
    model_config = BASE_CONFIG

    def __getitem__(self, item: str):
        return getattr(self, item)


    @property
    def config_name(self) -> str:
        return type(self).__name__

    def shallow_dump(self):
        fields = chain(self.__class__.model_fields.items(),
                       self.model_computed_fields.items())
        return {
            field[1].serialization_alias or field[1].alias or field[0]: getattr(self, field[0]) for field in fields}

    def get(self, key: str, default: Any = None):
        try:
            return self[key]
        except AttributeError:
            return default



