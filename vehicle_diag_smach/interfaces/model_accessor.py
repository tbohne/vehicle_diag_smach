#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import Union

from tensorflow import keras


class ModelAccessor(ABC):
    """
    Interface that defines the state machine's access to the model server.
    """

    @abstractmethod
    def get_model_by_component(self, component: str) -> Union[keras.models.Model, None]:
        """
        Retrieves a trained model to classify signals of the specified vehicle component.

        :param component: vehicle component to retrieve trained model for
        :return: trained model or `None` if unavailable
        """
        pass
