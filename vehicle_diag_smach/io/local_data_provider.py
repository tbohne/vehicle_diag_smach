#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List

from PIL import Image

from vehicle_diag_smach.data_types.intermediate_results import IntermediateResults
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class LocalDataProvider(DataProvider):
    """
    Implementation of the data provider interface.
    """

    def __init__(self):
        pass

    def provide_intermediate_results(self, intermediate_results: IntermediateResults) -> None:
        """
        Provides intermediate results to the hub UI.

        :param intermediate_results: intermediate results to be displayed on hub UI
        """
        pass

    def provide_causal_graph_visualizations(self, visualizations: List[Image.Image]) -> None:
        """
        Provides causal graph visualizations to the hub UI.

        :param visualizations: causal graph visualizations to be displayed on hub UI
        """
        for vis in visualizations:
            vis.show()
