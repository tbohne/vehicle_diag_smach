#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne


import numpy as np

from vehicle_diag_smach.interfaces.rule_based_model import RuleBasedModel


class Saugrohrdrucksensor(RuleBasedModel):
    """
    Rule-based model for "Saugrohrdrucksensor" channels.
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
        if chan_name == "Signalleitung (Druck) des Saugrohrdrucksensors":
            return False
        elif chan_name == "Signalleitung (Temperatur) des Saugrohrdrucksensors":
            return False
        elif chan_name == "Masseleitung des Saugrohrdrucksensors":
            return False
        elif chan_name == "Versorgungsspannung des Saugrohrdrucksensors":
            return False
        else:
            # TODO: default behavior for unknown chan
            return False
