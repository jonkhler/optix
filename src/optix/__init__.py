from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax


__all__ = ["Lens", "BoundLens", "FreeLens", "BoundLens", "Focused", "focus"]




class Lens[T, S](Protocol):
    """ A protocol for lenses. """

    def get(self) -> S:
        """ Get the value of the focus in the object. """
        ...

    def set(self, val: S) -> T:
        """ Set the value of the focus in the object. """
        ...

    def apply(self, update: Callable[[S], S]) -> T:
        """ Apply a function to the focused value in the object. """
        ...



class FreeLens[T, S](eqx.Module):
    """ A lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.
    """

    where: Callable[[T], S]
    
    def bind(self, obj: T) -> BoundLens[T, S]:
        """ Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return BoundLens(obj, self.where)


class BoundLens[T, S](eqx.Module):
    """ A lens that focuses on a value in a bound object.

    Args:
        obj: The object to focus on. 
        where: A function that retrieves the focused value from the object.
    """

    obj: T
    where: Callable[[T], S]

    def get(self) -> S:
        """ Get the value of the focus in the object.

        Returns:
            The focused value.
        """
        return self.where(self.obj)
    
    def set(self, val: S) -> T:
        """ Set the value of the focus in the object.

        Args:
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=val)
    
    def apply(self, update: Callable[[S], S]) -> T:
        """ Apply a function to the focused value in the object.

        Args:
            update: The function to apply to the focused value.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=update(self.get()))
    


class ArrayLens[T, I](eqx.Module):

    lens: Lens[T, jax.Array]
    index: I

    def get(self, **kwargs) -> jax.Array:
        arr = self.lens.get()
        return arr.at[self.index].get(**kwargs)
    
    def set(self, val: jax.Array, **kwargs) -> T:
        return self.lens.apply(lambda arr: arr.at[self.index].set(val, **kwargs))
    
    def apply(self, update: Callable[[jax.Array], jax.Array], **kwargs) -> T:
        return self.lens.apply(lambda arr: arr.at[self.index].apply(update, **kwargs))



class Focused[T](eqx.Module):
    """ An object that can be focused on.

    Args:
        obj: The object to focus on.
    """
    obj: T

    def at[S](self, where: Callable[[T], S]) -> Lens[T, S]:
        """ Focus on a value in the object.

        Args:
            where: A function that retrieves the focused value from the object.

        Returns:
            A bound lens.
        """
        return BoundLens(self.obj, where)

    def at_index[I](self, where: Callable[[T], jax.Array], index: I) -> ArrayLens[T, jax.Array]:
        """ Focus on an index in an array in the object.

        Args:
            where: A function that retrieves the focused value from the object.
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        return ArrayLens(self.at(where), index)


def focus[T](obj: T) -> Focused[T]:
    """ Focus on an object. """
    return Focused(obj)