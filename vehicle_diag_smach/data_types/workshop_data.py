#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class WorkshopData:
    """
    Represents workshop data, which is communicated to the state machine.
    """

    def __init__(self, num_of_parallel_rec: int):
        """
        Inits the workshop data.

        :param num_of_parallel_rec: maximum number of parallel recordings based on workshop equipment
        """
        self.num_of_parallel_rec = num_of_parallel_rec
