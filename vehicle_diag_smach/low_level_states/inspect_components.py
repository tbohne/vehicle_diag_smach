#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored


class InspectComponents(smach.State):
    """
    State in the low-level SMACH representing situations where manual inspection of suspect components, for which
    oscilloscope diagnosis is not appropriate, is performed.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['no_anomaly', 'detected_anomalies', 'no_anomaly_no_more_comp'],
                             input_keys=['suggestion_list'],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'INSPECT_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("no_anomaly" | "detected_anomalies" | "no_anomaly_no_more_comp")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("INSPECT_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        print("SUGGESTION LIST:", userdata.suggestion_list)
        # TODO: to be implemented
        no_anomaly = True
        # TODO: are there remaining suspect components? (atm every component is suggested each case)
        no_more_measuring_pos = True

        if no_anomaly and no_more_measuring_pos:
            return "no_anomaly_no_more_comp"
        elif no_anomaly:
            return "no_anomaly"
        return "detected_anomalies"
