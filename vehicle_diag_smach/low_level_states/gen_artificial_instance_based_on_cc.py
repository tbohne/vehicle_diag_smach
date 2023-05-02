#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored


class GenArtificialInstanceBasedOnCC(smach.State):
    """
    State in the low-level SMACH that represents situations in which an artificial DTC instance is generated
    based on the customer complaints. Used for cases where no OBD information is available.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['generated_artificial_instance'],
                             input_keys=['customer_complaints'],
                             output_keys=['generated_instance'])

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
        return "generated_artificial_instance"
