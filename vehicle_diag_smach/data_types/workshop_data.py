#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from datetime import date

from typing import Dict, Union


class WorkshopData:
    """
    Represents workshop data communicated to the state machine.
    """

    def __init__(self, num_of_parallel_rec: int, diag_date: date) -> None:
        """
        Inits the workshop data.

        :param num_of_parallel_rec: maximum number of parallel recordings based on workshop equipment
        :param diag_date: date of the diagnosis process
        """
        self.num_of_parallel_rec = num_of_parallel_rec
        self.diag_date = diag_date

    def get_json_representation(self) -> Dict[str, Union[int, date]]:
        """
        Returns a JSON representation of the workshop data.

        :return: JSON representation of workshop data
        """
        return {
            "max_num_of_parallel_rec": self.num_of_parallel_rec,
            "diag_date": self.diag_date
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the workshop data.

        :return: string representation of workshop data
        """
        return "num of parallel rec: " + str(self.num_of_parallel_rec) + ", diag date: " + str(self.diag_date)
