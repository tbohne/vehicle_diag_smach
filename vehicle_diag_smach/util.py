#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from tensorflow import keras


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
