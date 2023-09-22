#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List

from PIL import Image
from termcolor import colored

from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class TestDataProvider(DataProvider):
    """
    Implementation of the data provider interface used for different unit test scenarios.
    """

    def __init__(self) -> None:
        pass

    def provide_causal_graph_visualizations(self, visualizations: List[Image.Image]) -> None:
        """
        Provides causal graph visualizations to the hub UI.

        :param visualizations: causal graph visualizations to be displayed on hub UI
        """
        pass

    def provide_heatmaps(self, heatmaps: Image, title: str) -> None:
        """
        Provides heatmap visualizations to the hub UI.

        :param heatmaps: heatmap visualizations to be displayed on hub UI
        :param title: title of the heatmap plot (component + result of classification + score)
        """
        pass

    def provide_diagnosis(self, fault_paths: List[str]) -> None:
        """
        Provides the final diagnosis in the form of a set of fault paths to the hub UI.

        :param fault_paths: final diagnosis to be displayed on hub UI
        """
        for fault_path in fault_paths:
            print(colored(fault_path, "red", "on_white", ["bold"]))

    def provide_state_transition(self, state_transition: StateTransition) -> None:
        """
        Provides a transition performed by the state machine as part of a diagnostic process.

        :param state_transition: state transition (prev state -- (transition link) --> current state)
        """
        print("-----------------------------------------------------------")
        print("Performed state transition:", state_transition)
        print("-----------------------------------------------------------")
