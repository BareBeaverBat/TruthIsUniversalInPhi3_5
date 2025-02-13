import jaxtyping
import torch
from beartype import beartype
from numpy._typing import NDArray


def is_binary(labels: torch.Tensor | NDArray) -> bool:
    result = ((labels == 1) | (labels == 0)).all()
    return result.item() if isinstance(result, torch.Tensor) else result


def is_bipolar(labels: torch.Tensor | NDArray) -> bool:
    abs_vals = labels.abs() if isinstance(labels, torch.Tensor) else abs(labels)
    result = (abs_vals == 1).all()
    return result.item() if isinstance(result, torch.Tensor) else result


# annotating a function with this ensures that the beartype runtime type checking of arguments, return values, _and_
#  local variable assignments will make use of a Jaxtyping dynamic context which keeps track of the current values
#  of various names for array dimension sizes in the current function call. Without this, much of the "tensors have
#  shapes that are consistent with each other" benefit of jaxtyping/beartype will not be realized
bear_jax_typed = jaxtyping.jaxtyped(typechecker=beartype)
