from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx


__all__ = ["Lens", "FreeLens", "Lens", "Focused", "focus"]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


class FreeLens(eqx.Module, Generic[T, S]):
    """ A lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.
    """

    where: Callable[[T], S]

    def get(self, obj: T) -> S:
        """ Get the value of the focus in the object.

        Args:
            obj: The object to query.
        
        Returns:
            The focused value.
        """
            
        return self.where(obj)
    
    def set(self, obj: T, val: S) -> T:
        """ Set the value of the focus in the object.

        Args:
            obj: The object to modify.
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, obj, val)
    
    def apply(self, obj: T, update: Callable[[S], S]) -> T:
        """ Apply a function to the focused value in the object.

        Args:
            update: The object to modify.
            new: The function to apply to the focused value.
            
        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, obj, replace=update(self.get(obj)))
    
    def bind(self, obj: T) -> Lens[T, S]:
        """ Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return Lens(obj, self.where)


class Lens(eqx.Module, Generic[T, S]):
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
    

class Focused(eqx.Module, Generic[T]):
    """ An object that can be focused on.

    Args:
        obj: The object to focus on.
    """
    obj: T

    def at(self, where: Callable[[T], S]) -> Lens[T, S]:
        """ Focus on a value in the object.

        Args:
            where: A function that retrieves the focused value from the object.

        Returns:
            A bound lens.
        """
        return Lens(self.obj, where)


def focus(obj: T) -> Focused[T]:
    """ Focus on an object. """
    return Focused(obj)