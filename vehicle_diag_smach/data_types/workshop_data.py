#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from datetime import date


class WorkshopData:
    """
    Represents workshop data, which is communicated to the state machine.
    """

    def __init__(self, num_of_parallel_rec: int, diag_date: date):
        """
        Inits the workshop data.

        :param num_of_parallel_rec: maximum number of parallel recordings based on workshop equipment
        :param diag_date: date of the diagnosis process
        """
        self.num_of_parallel_rec = num_of_parallel_rec
        self.diag_date = diag_date

    def __str__(self):
        return "num of parallel rec: " + str(self.num_of_parallel_rec) + ", diag date: " + str(self.diag_date)
