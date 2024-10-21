#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod

import numpy as np


class RuleBasedModel(ABC):
    """
    Interface for a rule-based evaluation model for oscillogram signals.
    """

    @abstractmethod
    def predict(self, input_signal: np.ndarray) -> bool:
        """
        Predicts whether the input signal comprises an anomaly.

        :param input_signal: input signal to be classified
        :return: true -> anomaly, false -> no anomaly
        """
        pass
