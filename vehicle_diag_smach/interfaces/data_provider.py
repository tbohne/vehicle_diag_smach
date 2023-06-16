#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod

from vehicle_diag_smach.data_types.intermediate_results import IntermediateResults


class DataProvider(ABC):
    """
    Interface that defines the state machine's provision of intermediate results to be displayed in the hub UI.
    """

    @abstractmethod
    def provide_intermediate_results(self) -> IntermediateResults:
        """
        Provides intermediate results to the hub UI.

        :return: intermediate results
        """
        pass
