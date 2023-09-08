#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProcCustomerComplaints(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic enters the customer complaints
    to the processing system (fault tree, decision tree, XPS, ...).
    """

    def __init__(self, data_accessor: DataAccessor, data_provider: DataProvider) -> None:
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self, outcomes=['received_complaints', 'no_complaints'], input_keys=[''], output_keys=[''])
        self.data_accessor = data_accessor
        self.data_provider = data_provider

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROC_CUSTOMER_COMPLAINTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROC_CUSTOMER_COMPLAINTS' state.

        :param userdata: input of the state (provided user data)
        :return: outcome of the state ("received_complaints" | "no_complaints")
        """
        self.log_state_info()
        customer_complaints = self.data_accessor.get_customer_complaints()
        if customer_complaints.root != "":
            print("customer complaints session:")
            customer_complaints.print_all_info()
            print("customer XPS session protocol saved..")
            self.data_provider.provide_state_transition(StateTransition(
                "PROC_CUSTOMER_COMPLAINTS", "READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "received_complaints"
            ))
            # TODO: when we know more about the precise structure etc. of CC, we should enter them into the KG here
            return "received_complaints"
        else:
            print("starting diagnosis without customer complaints..")
            self.data_provider.provide_state_transition(StateTransition(
                "PROC_CUSTOMER_COMPLAINTS", "READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "no_complaints"
            ))
            return "no_complaints"
