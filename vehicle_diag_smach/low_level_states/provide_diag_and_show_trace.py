#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideDiagAndShowTrace(smach.State):
    """
    State in the low-level SMACH that represents situations in which the diagnosis is provided in combination with
    a detailed trace of all the relevant information that lead to it. Additionally, the diagnosis is entered into the
    knowledge graph.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['uploaded_diag'],
                             input_keys=['diagnosis'],
                             output_keys=[''])
        self.data_provider = data_provider

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        :param userdata: input of state
        :return: outcome of the state ("uploaded_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_DIAG_AND_SHOW_TRACE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # TODO: enter diagnosis into KG
        #   - it's important to log the whole context - everything that could be meaningful
        #   - in the long run, this is where we collect the data that we initially lacked, e.g., for automated
        #     data-driven RCA
        #   - to be logged (diagnosis together with):
        #       - associated symptoms, DTCs, components (distinguishing root causes and side effects) etc.
        # TODO: show diagnosis + trace

        fault_paths = []
        for key in userdata.diagnosis.keys():
            print("\nidentified anomalous component:", key)
            print("fault path:")
            path = userdata.diagnosis[key][::-1]
            path = [path[i] if i == len(path) - 1 else path[i] + " -> " for i in range(len(path))]
            fault_paths.append("".join(path))
        self.data_provider.provide_diagnosis(fault_paths)
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_DIAG_AND_SHOW_TRACE", "diag", "uploaded_diag"
        ))
        return "uploaded_diag"
