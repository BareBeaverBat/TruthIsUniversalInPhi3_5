import torch
from numpy._typing import NDArray


def is_binary(labels: torch.Tensor | NDArray) -> bool:
    result = ((labels == 1) | (labels == 0)).all()
    return result.item() if isinstance(result, torch.Tensor) else result


def is_bipolar(labels: torch.Tensor | NDArray) -> bool:
    abs_vals = labels.abs() if isinstance(labels, torch.Tensor) else abs(labels)
    result = (abs_vals == 1).all()
    return result.item() if isinstance(result, torch.Tensor) else result
