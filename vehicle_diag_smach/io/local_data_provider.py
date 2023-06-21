#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os
import platform
from typing import List

from PIL import Image
from termcolor import colored

from vehicle_diag_smach.data_types.intermediate_results import IntermediateResults
from vehicle_diag_smach.data_types.state_transition import StateTransition
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

    def provide_heatmaps(self, heatmaps: Image, title: str) -> None:
        """
        Provides heatmap visualizations to the hub UI.

        :param heatmaps: heatmap visualizations to be displayed on hub UI
        :param title: title of the heatmap plot (component + result of classification + score)
        """
        title = title.replace(" ", "_") + ".png"
        heatmaps.save(title)
        # determine platform and open file with default image viewer
        if platform.system() == "Windows":
            os.system("start " + title)
        elif platform.system() == "Darwin":  # macOS
            os.system("open " + title)
        else:  # Linux
            os.system("xdg-open " + title)

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
