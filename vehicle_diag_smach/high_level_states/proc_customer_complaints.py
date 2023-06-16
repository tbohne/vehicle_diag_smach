#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.interfaces.data_accessor import DataAccessor


class ProcCustomerComplaints(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic enters the customer complaints
    to the processing system (fault tree, decision tree, XPS, ...).
    """

    def __init__(self, data_accessor: DataAccessor):
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        """
        smach.State.__init__(self, outcomes=['received_complaints', 'no_complaints'], input_keys=[''], output_keys=[''])
        self.data_accessor = data_accessor

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROC_CUSTOMER_COMPLAINTS' state.

        :param userdata: input of state (provided user data)
        :return: outcome of the state ("received_complaints" | "no_complaints")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROC_CUSTOMER_COMPLAINTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        customer_complaints = self.data_accessor.get_customer_complaints()
        if customer_complaints.root != "":
            print("customer complaints session:")
            customer_complaints.print_all_info()
            print("customer XPS session protocol saved..")
            return "received_complaints"
        else:
            print("starting diagnosis without customer complaints..")
            return "no_complaints"
