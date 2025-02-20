from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax

__all__ = ["Lens", "FreeLens", "focus"]


@dataclass(frozen=True)
class _LensIndexingHelper[T, S]:
    lens: Lens[T, S]

    def __getitem__(self, index) -> Lens[T, S]:
        return _ArrayLens(self.lens, index)


@dataclass(frozen=True)
class _FreeLensIndexingHelper[T, S]:
    lens: FreeLens[T, S]

    def __getitem__(self, index) -> FreeLens[T, S]:
        return _FreeArrayLens(self.lens, index)


class Lens[T, S](Protocol):
    """A protocol for lenses."""

    def get(self) -> S:
        """Get the value of the focus in the object."""
        ...

    def set(self, val: S) -> T:
        """Set the value of the focus in the object."

        Args:
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        ...

    def apply(self, update: Callable[[S], S]) -> T:
        """Apply a function to the focused value in the object.

        Args:
            update: The function to apply to the focused value.

        Returns:
            A modified copy of the object
        """
        ...

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        ...


class FreeLens[T, S](Protocol):
    """A protocol for free lenses."""

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        ...

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        ...


@dataclass(frozen=True)
class _FreeLens[T, S]:
    """A lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.
    """

    where: Callable[[T], S]

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return _BoundLens(obj, self.where)

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        return _FreeLensIndexingHelper(self)


@dataclass(frozen=True)
class _BoundLens[T, S]:
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

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        return _LensIndexingHelper(self)


@dataclass(frozen=True)
class _ArrayLens[T, S, I]:
    """A lens that focuses on an index in an array in a bound object.

    Args:
        lens: The lens to focus on.
        index: The index to focus on.
    """

    lens: Lens[T, S]
    index: I

    def get(self, **kwargs) -> S:
        """Get the value of the focus in the object.

        Args:
            **kwargs: Additional arguments to pass to the jax getter.

        Returns:
            The focused value.
        """

        def _getter(x: jax.Array) -> jax.Array:
            return x.at[self.index].get(**kwargs)

        return jax.tree.map(_getter, self.lens.get())

    def set(self, val: S, **kwargs) -> T:
        def _setter(x: jax.Array, y: jax.Array) -> jax.Array:
            return x.at[self.index].set(y, **kwargs)

        return self.lens.apply(lambda out: jax.tree.map(_setter, out, val))

    def apply(self, update: Callable[[S], S], **kwargs) -> T:
        def _update(x: jax.Array) -> jax.Array:
            return x.at[self.index].apply(update, **kwargs)

        return self.lens.apply(lambda out: jax.tree.map(_update, out))

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        raise ValueError("Cannot index and already indexed lens.")


@dataclass(frozen=True)
class _FreeArrayLens[T, S, I]:
    """A lens that focuses on an index in an array in a bound object.

    Args:
        lens: The lens to focus on.
        index: The index to focus on.
    """

    lens: _FreeLens[T, S]
    index: I

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return _ArrayLens(self.lens.bind(obj), self.index)

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        raise ValueError("Cannot index and already indexed lens.")


@dataclass(frozen=True)
class _Focused[T]:
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
        return _BoundLens(self.obj, where)


def focus[T](obj: T) -> _Focused[T]:
    """Focus on an object."""
    return _Focused(obj)


def lens[T, S](where: Callable[[T], S]) -> _FreeLens[T, S]:
    """Create a lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.

    Returns:
        An unbound lens.
    """
    return _FreeLens(where)
