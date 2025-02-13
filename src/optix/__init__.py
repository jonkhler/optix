from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax

__all__ = ["Lens", "BoundLens", "UnboundLens", "BoundLens", "Focused", "focus"]


class Lens[T, S](Protocol):
    """A protocol for lenses."""

    def get(self) -> S:
        """Get the value of the focus in the object."""
        ...

    def set(self, val: S) -> T:
        """Set the value of the focus in the object."""
        ...

    def apply(self, update: Callable[[S], S]) -> T:
        """Apply a function to the focused value in the object."""
        ...


class FreeLens[T, S](Protocol):
    """A protocol for free lenses."""

    def bind(self, obj: T) -> BoundLens[T, S]:
        """Bind the lens to an object."""
        ...


class UnboundLens[T, S](eqx.Module):
    """A lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.
    """

    where: Callable[[T], S]

    def bind(self, obj: T) -> BoundLens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return BoundLens(obj, self.where)


class BoundLens[T, S](eqx.Module):
    """A lens that focuses on a value in a bound object.

    Args:
        obj: The object to focus on.
        where: A function that retrieves the focused value from the object.
    """

    obj: T
    where: Callable[[T], S]

    def get(self) -> S:
        """Get the value of the focus in the object.

        Returns:
            The focused value.
        """
        return self.where(self.obj)

    def set(self, val: S) -> T:
        """Set the value of the focus in the object.

        Args:
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=val)

    def apply(self, update: Callable[[S], S]) -> T:
        """Apply a function to the focused value in the object.

        Args:
            update: The function to apply to the focused value.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=update(self.get()))


class BoundArrayLens[T, S, I](eqx.Module):
    lens: Lens[T, S]
    index: I

    def get(self, **kwargs) -> S:
        return jax.tree.map(lambda x: x.at[self.index].get(**kwargs), self.lens.get())

    def set(self, val: S, **kwargs) -> T:
        return self.lens.apply(
            lambda out: jax.tree.map(
                lambda x, y: x.at[self.index].set(y, **kwargs), out, val
            )
        )

    def apply(self, update: Callable[[S], S], **kwargs) -> T:
        return self.lens.apply(
            lambda out: jax.tree.map(
                lambda x: x.at[self.index].apply(update, **kwargs), out
            )
        )


class UnboundArrayLens[T, S, I](eqx.Module):
    lens: UnboundLens[T, S]
    index: I

    def bind(self, obj: T) -> BoundArrayLens[T, S, I]:
        return BoundArrayLens(self.lens.bind(obj), self.index)


class Focused[T](eqx.Module):
    """An object that can be focused on.

    Args:
        obj: The object to focus on.
    """

    obj: T

    def at[S](self, where: Callable[[T], S]) -> Lens[T, S]:
        """Focus on a value in the object.

        Args:
            where: A function that retrieves the focused value from the object.

        Returns:
            A bound lens.
        """
        return BoundLens(self.obj, where)

    def at_index[S, I](self, where: Callable[[T], S], index: I) -> BoundArrayLens[T, S]:
        """Focus on an index in an array in the object.

        Args:
            where: A function that retrieves the focused value from the object.
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        return BoundArrayLens(self.at(where), index)


def focus[T](obj: T) -> Focused[T]:
    """Focus on an object."""
    return Focused(obj)


def lens[T, S](where: Callable[[T], S]) -> UnboundLens[T, S]:
    """Create a lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.

    Returns:
        An unbound lens.
    """
    return UnboundLens(where)


def array_lens[T, S, I](where: Callable[[T], S], index: I) -> UnboundArrayLens[T, S, I]:
    """Create a lens that focuses on an index in an array in an object.

    Args:
        where: A function that retrieves the focused value from the object.
        index: The index to focus on.

    Returns:
        An unbound lens.
    """
    return UnboundArrayLens(lens(where), index)
