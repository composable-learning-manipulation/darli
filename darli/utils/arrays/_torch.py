from dataclasses import dataclass
from ._arraylike import ArrayLike, ArrayLikeFactory
from typing import Union
import torch


@dataclass
class TorchLike(ArrayLike):
    """Class wrapping Torch types."""

    array: torch.Tensor

    def __setitem__(self, idx, value: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides set item operator."""
        if isinstance(value, TorchLike):
            self.array[idx] = value.array.reshape(self.array[idx].shape)
        else:
            self.array[idx] = value

    def __getitem__(self, idx) -> "TorchLike":
        """Overrides get item operator."""
        return TorchLike(self.array[idx])

    def to(self, dev: torch.device) -> "TorchLike":
        self.array.to(dev)
        return self

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self) -> "TorchLike":
        """
        Returns:
            TorchLike: transpose of the array
        """
        return TorchLike(self.array.T)

    def __matmul__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides @ operator."""
        if isinstance(other, TorchLike):
            return TorchLike(self.array @ other.array)
        else:
            return TorchLike(self.array @ torch.tensor(other))

    def __rmatmul__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides @ operator."""
        if isinstance(other, TorchLike):
            return TorchLike(other.array @ self.array)
        else:
            return TorchLike(other @ self.array)

    def __mul__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides * operator."""
        if isinstance(other, TorchLike):
            return TorchLike(self.array * other.array)
        else:
            return TorchLike(self.array * other)

    def __rmul__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides * operator."""
        if isinstance(other, TorchLike):
            return TorchLike(other.array * self.array)
        else:
            return TorchLike(other * self.array)

    def __truediv__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides / operator."""
        if isinstance(other, TorchLike):
            return TorchLike(self.array / other.array)
        else:
            return TorchLike(self.array / other)

    def __add__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides + operator."""
        if not isinstance(other, TorchLike):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides + operator."""
        if not isinstance(other, TorchLike):
            return TorchLike(self.array + other)
        return TorchLike(self.array + other.array)

    def __sub__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides - operator"""
        if not isinstance(other, TorchLike):
            return TorchLike(self.array.squeeze() - other.squeeze())
        return TorchLike(self.array.squeeze() - other.array.squeeze())

    def __rsub__(self, other: Union["TorchLike", torch.Tensor]) -> "TorchLike":
        """Overrides - operator"""
        if not isinstance(other, TorchLike):
            return TorchLike(other.squeeze() - self.array.squeeze())
        return TorchLike(other.array.squeeze() - self.array.squeeze())

    def __neg__(self):
        """Overrides - operator"""
        return TorchLike(-self.array)


class TorchLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x) -> "TorchLike":
        """
        Returns:
            TorchLike: zero matrix of dimension x
        """
        return TorchLike(torch.zeros(*x))

    @staticmethod
    def eye(x: int) -> "TorchLike":
        """
        Args:
            x (int): matrix dimension

        Returns:
            TorchLike: Identity matrix of dimension x
        """
        return TorchLike(torch.eye(x))

    @staticmethod
    def array(x) -> "TorchLike":
        """
        Returns:
            TorchLike: Vector wrapping *x
        """
        return TorchLike(torch.tensor(x))

    @staticmethod
    def norm_2(x) -> "TorchLike":
        """
        Returns:
            TorchLike: Norm of x
        """
        return TorchLike(torch.linalg.norm(x))

    @staticmethod
    def solve(a: "TorchLike", b: "TorchLike") -> "TorchLike":
        """
        Args:
            a (TorchLike): Matrix
            b (TorchLike): Vector

        Returns:
            TorchLike: Solution of the linear system a @ x = b
        """
        return TorchLike(torch.linalg.solve(a, b))
