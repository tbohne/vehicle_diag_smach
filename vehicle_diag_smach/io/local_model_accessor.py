#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import Union, Tuple

from tensorflow import keras

from vehicle_diag_smach.config import TRAINED_MODEL_POOL
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class LocalModelAccessor(ModelAccessor):
    """
    Implementation of the model accessor interface using local model files.
    """

    def __init__(self):
        pass

    def get_keras_univariate_ts_classification_model_by_component(
            self, component: str
    ) -> Union[Tuple[keras.models.Model, dict], None]:
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
        try:
            trained_model_file = TRAINED_MODEL_POOL + component + ".h5"
            print("loading trained model:", trained_model_file)
            model_meta_info = {
                "normalization_method": "z-normalization",
                "model_id": "keras_univariate_ts_classification_model_001"
            }
            return keras.models.load_model(trained_model_file), model_meta_info
        except OSError as e:
            print("no trained model available for the signal (component) to be classified:", component)
            print("ERROR:", e)
