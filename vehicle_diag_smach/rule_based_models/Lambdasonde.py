#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne


import numpy as np

from vehicle_diag_smach.interfaces.rule_based_model import RuleBasedModel
from vehicle_diag_smach.rule_based_models.rule_based_signal_classification import classify_sensor_rule_based


class Lambdasonde(RuleBasedModel):
    """
    Rule-based model for "Lambdasonde" channels.
    """

    def __init__(self):
        pass

    def predict(self, input_signal: np.ndarray, chan_name: str = "") -> bool:
        """
        Predicts whether the input signal comprises an anomaly.

        :param input_signal: input signal to be classified
        :param chan_name: optional name of the channel
        :return: true -> anomaly, false -> no anomaly
        """
        if chan_name == "Masseleitung der Heizung der Lambdasonde":
            return classify_sensor_rule_based(input_signal, relevant_value=14., relevant_value_means_anomaly=False)
        elif chan_name == "Plusleitung der Heizung der Lambdasonde":
            return classify_sensor_rule_based(
                input_signal, relevant_value=0., relevant_value_means_anomaly=True, outlier_intolerance=0.2
            )
        elif chan_name == "Plusleitung der Lambdasonde":
            return classify_sensor_rule_based(
                input_signal, relevant_value=0., relevant_value_means_anomaly=True, outlier_intolerance=0.2
            )
        elif chan_name == "Masseleitung der Lambdasonde":
            return classify_sensor_rule_based(input_signal, relevant_value=0., relevant_value_means_anomaly=False)
        else:
            # TODO: default behavior for unknown chan
            return False
