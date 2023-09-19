#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import Dict, List, Tuple

import numpy as np
from oscillogram_classification import preprocess
from tensorflow import keras

from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData


def validate_keras_model(model: keras.models.Model) -> None:
    """
    The provided model is expected to be a Keras model satisfying the following assumptions:
        - input_shape: (None, len_of_ts, 1)
        - output_shape: (None, 1)
    Thus, in both cases we have a variable batch size due to `None`. For the input we expect a list of scalars and
    for the output exactly one scalar.

    :param model: model to be validated

    :raise ValueError: if the model shape doesn't match the expected one
    """
    in_shape = model.input_shape
    out_shape = model.output_shape
    expected_in_shape = (None, in_shape[1], 1)
    expected_out_shape = (None, 1)

    if len(in_shape) != len(expected_in_shape) or any(dim1 != dim2 for dim1, dim2 in zip(in_shape, expected_in_shape)):
        raise ValueError(f"Unexpected input shape - expected: {expected_in_shape}, got: {in_shape}")

    if len(out_shape) != len(expected_out_shape) \
            or any(dim1 != dim2 for dim1, dim2 in zip(out_shape, expected_out_shape)):
        raise ValueError(f"Unexpected output shape - expected: {expected_out_shape}, got: {out_shape}")


def preprocess_time_series_based_on_model_meta_info(
        model_meta_info: Dict[str, str], voltages: List[float]
) -> List[float]:
    """
    Preprocesses the time series based on model metadata (e.g. normalization method).
    The preprocessing always depends on the trained model that is going to be applied.
    Therefore, this kind of meta information has to be saved for each trained model.

    :param model_meta_info: metadata about the trained model (e.g. normalization method)
    :param voltages: raw input (voltage values)
    :return: preprocessed input (voltage values)
    """
    print("model meta info:", model_meta_info)
    if model_meta_info["normalization_method"] == "z_norm":
        return preprocess.z_normalize_time_series(voltages)
    elif model_meta_info["normalization_method"] == "min_max_norm":
        return preprocess.min_max_normalize_time_series(voltages)
    elif model_meta_info["normalization_method"] == "dec_norm":
        return preprocess.decimal_scaling_normalize_time_series(voltages, 2)
    elif model_meta_info["normalization_method"] == "log_norm":
        return preprocess.logarithmic_normalize_time_series(voltages, 10)
    return voltages


def no_trained_model_available(osci_data: OscillogramData, suggestion_list: Dict[str, Tuple[str, bool]]) -> None:
    """
    Handles cases where no trained model is available for the specified component.

    :param osci_data: oscillogram data
    :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
    """
    print("no trained model available for the signal (component) to be classified:", osci_data.comp_name)
    print("adding it to the list of components to be verified manually..")
    suggestion_list[osci_data.comp_name] = suggestion_list[osci_data.comp_name][0], False


def invalid_model(osci_data: OscillogramData, suggestion_list: Dict[str, Tuple[str, bool]], error: ValueError) -> None:
    """
    Handles cases where no valid trained model is available for the specified component.

    :param osci_data: oscillogram data
    :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
    :param error: thrown value error
    """
    print("invalid model for the signal (component) to be classified:", osci_data.comp_name)
    print("error:", error)
    print("adding it to the list of components to be verified manually..")
    suggestion_list[osci_data.comp_name] = suggestion_list[osci_data.comp_name][0], False


def construct_net_input(model: keras.models.Model, voltages: List[float]) -> np.ndarray:
    """
    Constructs / reshapes the input for the trained neural net model.

    :param model: trained neural net model
    :param voltages: input voltage values (time series) to be reshaped
    :return: constructed / reshaped input
    """
    net_input_size = model.layers[0].output_shape[0][1]
    assert net_input_size == len(voltages)
    net_input = np.asarray(voltages).astype('float32')
    return net_input.reshape((net_input.shape[0], 1))
