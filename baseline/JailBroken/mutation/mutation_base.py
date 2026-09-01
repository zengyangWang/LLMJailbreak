"""
Mutation Abstract Class
============================================
This module defines the abstract base class for mutation methods used in jailbreaking datasets.
These methods transform sequences of text to produce potential adversarial examples, aiding in testing
and strengthening machine learning models against adversarial attacks.

The MutationBase class serves as the foundation for defining specific mutation strategies.
Subclasses implementing specific mutation techniques should override the relevant methods to provide
custom behavior for generating mutated instances.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Any


__all__ = [
    "MutationBase",
    "Instance",
    "JailbreakDataset",
]


class Instance:
    """Lightweight replacement for the EasyJailbreak ``Instance`` object."""

    def __init__(self, **kwargs: Any) -> None:
        self.parents: List[Instance] = kwargs.pop("parents", [])
        self.children: List[Instance] = kwargs.pop("children", [])
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "query"):
            self.query = ""
        if not hasattr(self, "jailbreak_prompt"):
            self.jailbreak_prompt = None
        if not hasattr(self, "language"):
            self.language = "en"

    def copy(self) -> "Instance":
        preserved = {
            key: copy.deepcopy(value)
            for key, value in self.__dict__.items()
            if key not in {"parents", "children"}
        }
        cloned = Instance(**preserved)
        return cloned


class JailbreakDataset(list):
    """Simple list-based dataset replacement with a convenience ``add`` method."""

    def add(self, instance: Instance) -> None:
        self.append(instance)


class MutationBase(ABC):
    """
    An abstract base class for defining mutation strategies that transform a sequence of text to produce
    potential adversarial examples. This class provides the framework for implementing various types of
    text mutations for generating adversarial examples in jailbreak datasets.
    """

    def __call__(
        self,
        jailbreak_dataset: Iterable[Instance],
        *args,
        **kwargs,
    ) -> JailbreakDataset:
        """
        Applies the mutation method to a given jailbreak dataset, generating a new dataset of mutated instances.
        This method provides basic logic for recording parent-child relationships between instances.
        For common 1-to-n mutations, overriding the ``_get_mutated_instance`` method is sufficient.
        For other mutation types, directly overriding the ``__call__`` method is recommended.
        """
        new_dataset = []
        for instance in jailbreak_dataset:
            mutated_instance_list = self._get_mutated_instance(instance, *args, **kwargs)
            new_dataset.extend(mutated_instance_list)
        return JailbreakDataset(new_dataset)

    def _get_mutated_instance(self, instance: Instance, *args, **kwargs) -> List[Instance]:
        """
        Abstract method to be implemented in subclasses for mutating an instance to generate a list of mutated instances.
        If the mutation method typically generates one or more mutated instances for each input instance,
        this method should be overridden. Otherwise, the ``__call__`` method should be overridden.
        """
        raise NotImplementedError
