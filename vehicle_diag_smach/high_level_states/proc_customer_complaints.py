#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from py4j.java_gateway import JavaGateway
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, XPS_SESSION_FILE


class ProcCustomerComplaints(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic enters the customer complaints
    to the processing system (fault tree, decision tree, XPS, ...).
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['received_complaints', 'no_complaints'], input_keys=[''], output_keys=[''])

    @staticmethod
    def launch_customer_xps() -> str:
        """
        Launches the expert system that processes the customer complaints.
        """
        print("establish connection to customer XPS server..")
        gateway = JavaGateway()
        customer_xps = gateway.entry_point
        return customer_xps.demo("../" + SESSION_DIR + "/" + XPS_SESSION_FILE)

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
        val = ""
        while val != "0" and val != "1":
            val = input("\nstarting diagnosis with [0] / without [1] customer complaints")

        if val == "0":
            print("result of customer xps: ", self.launch_customer_xps())
            print("customer XPS session protocol saved..")
            return "received_complaints"
        else:
            print("starting diagnosis without customer complaints..")
            return "no_complaints"
