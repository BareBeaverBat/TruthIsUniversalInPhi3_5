import jaxtyping
import numpy as np
import torch
from beartype import beartype
from numpy._typing import NDArray


FloatLikeT = (float | np.floating)


def is_binary(labels: torch.Tensor | NDArray) -> bool:
    result = ((labels == 1) | (labels == 0)).all()
    return result.item() if isinstance(result, torch.Tensor) else result


def is_bipolar(labels: torch.Tensor | NDArray) -> bool:
    abs_vals = labels.abs() if isinstance(labels, torch.Tensor) else abs(labels)
    result = (abs_vals == 1).all()
    return result.item() if isinstance(result, torch.Tensor) else result


def bear_jax_typed_with_independent_calls(func):
    """
    annotating a function with this ensures that the beartype runtime type checking of arguments, return values, _and_
    local variable assignments will make use of a Jaxtyping dynamic context which keeps track of the current values
    of various names for array dimension sizes __in the current function call__. Without this, much of the "tensors have
    shapes that are consistent with each other" benefit of jaxtyping/beartype will not be realized

    :param func: a callable object
    :return: the callable object with better runtime type checking, particularly for tensors that have jaxtyping
                annotations
    """
    def wrapper(*args, **kwargs):
        # The jaxtyping context manager is needed so that the context is reset after each call.
        with jaxtyping.jaxtyped("context"):
            return jaxtyping.jaxtyped(func, typechecker=beartype)(*args, **kwargs)

    return wrapper


def greatest_power_of_two_below(n: int) -> int:
    if n < 2:
        raise ValueError("n must be at least 2")
    return 1 << ((n - 1).bit_length() - 1)


def float_eq(a: FloatLikeT | int, b: FloatLikeT | int, tol: float = 1e-8) -> bool:
    return abs(a - b) < tol
