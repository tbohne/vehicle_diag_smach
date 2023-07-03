#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import Union, Tuple

from tensorflow import keras


class ModelAccessor(ABC):
    """
    Interface that defines the state machine's access to the model server.
    """

    @abstractmethod
    def get_keras_univariate_ts_classification_model_by_component(
            self, component: str) -> Union[Tuple[keras.models.Model, dict], None]:
        """
        Retrieves a trained model to classify signals of the specified vehicle component.

        The provided model is expected to be a Keras model satisfying the following assumptions:
            - input_shape: (None, len_of_ts, 1)
            - output_shape: (None, 1)
        Thus, in both cases we have a variable batch size due to `None`. For the input we expect a list of scalars and
        for the output exactly one scalar.

        :param component: vehicle component to retrieve trained model for
        :return: trained model and model meta info dictionary or `None` if unavailable
        """
        pass
