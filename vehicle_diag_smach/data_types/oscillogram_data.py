#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List

import pandas as pd


class OscillogramData:
    """
    Represents (multivariate) oscillogram data communicated to the state machine.
    """

    def __init__(self, time_series: List[pd.DataFrame], comp_name: str) -> None:
        """
        Inits the oscillogram data.

        The list of voltage values should never be empty, at least two data points are expected.

        :param time_series: recorded (multivariate) signal (voltage values over time)
        :param comp_name: name of the component the signal belongs to
        """
        for chan in time_series:
            assert len(chan) >= 2
        self.time_series = time_series
        self.comp_name = comp_name
