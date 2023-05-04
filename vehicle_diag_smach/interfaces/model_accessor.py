#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod

from tensorflow import keras


class ModelAccessor(ABC):
    """
    Interface that defines the state machine's access to the model server.
    """

    @abstractmethod
    def get_model_by_component(self, component: str) -> keras.engine.training.Model:
        pass
