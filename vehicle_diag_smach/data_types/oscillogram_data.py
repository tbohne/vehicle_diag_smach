#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class OscillogramData:
    """
    Represents oscillogram data, which is communicated to the state machine.
    """

    def __init__(self, time_series: list[float]):
        """
        Inits the oscillogram data.

        :param time_series: recorded signal (voltage values over time)
        """
        self.time_series = time_series
