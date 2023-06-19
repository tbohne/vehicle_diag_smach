#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

from vehicle_diag_smach.data_types.intermediate_results import IntermediateResults


class DataProvider(ABC):
    """
    Interface that defines the state machine's provision of intermediate results to be displayed in the hub UI.
    """

    @abstractmethod
    def provide_intermediate_results(self, intermediate_results: IntermediateResults) -> None:
        """
        Provides intermediate results to the hub UI.

        :param intermediate_results: intermediate results to be displayed on hub UI
        """
        pass

    def provide_causal_graph_visualizations(self, visualizations: List[Image]) -> None:
        """
        Provides causal graph visualizations to the hub UI.

        :param visualizations: causal graph visualizations to be displayed on hub UI
        """
        pass
