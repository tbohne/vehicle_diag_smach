#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Patricia Windler

import numpy as np
import pandas as pd


def classify_sensor_rule_based(signal: np.array, relevant_value: float, relevant_value_means_anomaly: bool,
                               threshold: float = 0.5, outlier_intolerance: float = 0.99):
    """
    Classifies whether a signal contains an anomaly, based on a rule which specifies which values the signal should
    take or not take.

    :param signal: 1D time series that should be classified
    :param relevant_value: value that the signal should either match or avoid
    :param relevant_value_means_anomaly: flag specifying whether the signal matching relevant_value or
        avoiding relevant_value indicates an anomaly
    :param threshold: defines a range around relevant_value within which the signal counts as "near the relevant
        value"
    :param outlier_intolerance: ratio specifying how many of the data points must / must not be "near the relevant
        value" (allowing for a small number of outliers) in order for the signal to count as regular
    :return: True if an anomaly has been found, else False
    """
    signal = signal.squeeze()
    assert (len(signal.shape) == 1)

    n_data_points_within_range = np.count_nonzero(np.abs(np.array(signal) - relevant_value) < threshold)

    if relevant_value_means_anomaly:
        signal_resembles_relevant_value = (len(signal) - n_data_points_within_range) / len(
            signal) >= outlier_intolerance
    else:
        signal_resembles_relevant_value = n_data_points_within_range / len(
            signal) >= outlier_intolerance

    return not signal_resembles_relevant_value


def check_for_jumps(signal, threshold: float = 1., offset: int = 20):
    """
    Checks for jumps in the signal.

    Peaks consisting of one time point are disregarded by applying the rolling mean first. Jumps are defined as
    differences in the signal value within a certain number of time points (corresponding to the maximal offset).

    :param signal: 1D time series that should be checked for jumps
    :param threshold: the threshold which defines a difference in signal value as significant
    :param offset: The offsets defines how many time steps the time points whose value difference are calculated
        may be apart, i.e. maximal length of a jump
    :return: whether at least one jump has been found
    """
    signal = signal.squeeze()
    assert (len(signal.shape) == 1)
    window_size = 5

    signal_df = pd.DataFrame(signal)

    rolling_mean = signal_df.rolling(window=window_size).mean()

    # calculate the maximum difference between time values that are at most -offset- time steps apart
    max_difference = np.max([rolling_mean.diff(periods=offset).abs().max() for offset in range(offset)])

    return max_difference > threshold
