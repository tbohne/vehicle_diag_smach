#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from tensorflow import keras

from vehicle_diag_smach.config import TRAINED_MODEL_POOL
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class LocalModelAccessor(ModelAccessor):
    """
    Implementation of the model accessor interface using local model files.
    """

    def __init__(self):
        pass

    def get_model_by_component(self, component: str) -> keras.models.Model:
        """
        Retrieves a trained classification model for the specified vehicle component.

        :param component: vehicle component to retrieve trained model for
        :return:
        """
        try:
            trained_model_file = TRAINED_MODEL_POOL + component + ".h5"
            print("loading trained model:", trained_model_file)
            return keras.models.load_model(trained_model_file)
        except OSError as e:
            print("no trained model available for the signal (component) to be classified:", component)
            print("ERROR:", e)
