#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class GenArtificialInstanceBasedOnCC(smach.State):
    """
    State in the low-level SMACH that represents situations in which an artificial DTC instance is generated
    based on the customer complaints. Used for cases where no OBD information is available.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['generated_artificial_instance'],
                             input_keys=['customer_complaints'],
                             output_keys=['generated_instance'])
        self.data_provider = data_provider

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC' state.

        :param userdata: input of state
        :return: outcome of the state ("generated_artificial_instance")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")
        print("CC:", userdata.customer_complaints)
        # TODO: generate instance based on provided CC
        userdata.generated_instance = "P2563"
        self.data_provider.provide_state_transition(StateTransition(
            "GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC", "SUGGEST_SUSPECT_COMPONENTS", "generated_artificial_instance"
        ))
        return "generated_artificial_instance"
