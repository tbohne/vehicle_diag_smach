#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideInitialHypothesisAndLogContext(smach.State):
    """
    State in the low-level SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self, outcomes=['no_diag'], input_keys=[''], output_keys=[''])
        self.data_provider = data_provider

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT' state.

        :param userdata: input of state
        :return: outcome of the state ("no_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "yellow", "on_grey", ["bold"]),
              "state..")
        print("############################################")
        # TODO: create log file for the failed diagnostic process to improve future diagnosis (missing knowledge etc.)
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "refuted_hypothesis", "no_diag"
        ))
        return "no_diag"
