#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List


class OscillogramData:
    """
    Represents oscillogram data, which is communicated to the state machine.
    """

    def __init__(self, time_series: List[float], comp_name: str):
        """
        Inits the oscillogram data.

        :param time_series: recorded signal (voltage values over time)
        :param comp_name: name of the component the signal belongs to
        """
        self.time_series = time_series
        self.comp_name = comp_name
