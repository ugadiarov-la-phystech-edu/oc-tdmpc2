import os
import random
from contextlib import contextmanager
from typing import Optional, Callable

import numpy as np
import torch


def set_random_seed(seed):
    """ set random seeds for all possible random libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_device(device_option: Optional[str]) -> torch.device:
    if device_option is None:
        device_option = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_option)


def resolve_compile_backend():
    try:
        import triton
        return 'inductor'
    except ImportError:
        pass

    try:
        import torch_xla
        return 'openxla'
    except ImportError:
        pass

    if torch.onnx.is_onnxrt_backend_supported():
        return 'onnxrt'
    return None


def try_torch_compile(fn: Callable):
    backend = resolve_compile_backend()
    if backend is None:
        return fn
    return torch.compile(fn, backend=backend)


def fix_numpy():
    import numpy
    setattr(numpy, "bool", bool)
    setattr(numpy, "float", float)


@contextmanager
def fix_posix_path():
    temp = None
    try:
        if os.name == 'nt':
            import pathlib
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        if os.name  == 'nt':
            pathlib.PosixPath = temp