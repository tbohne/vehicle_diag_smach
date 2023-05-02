#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, DTC_TMP_FILE, CC_TMP_FILE


class SelectBestUnusedErrorCodeInstance(smach.State):
    """
    State in the low-level SMACH that represents situations in which a best-suited, unused DTC instance is
    selected for further processing.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['selected_matching_instance(OBD_CC)', 'no_matching_selected_best_instance',
                                       'no_instance', 'no_instance_and_CC_already_used'],
                             input_keys=[''],
                             output_keys=['selected_instance', 'customer_complaints'])

    @staticmethod
    def remove_dtc_instance_from_tmp_file(remaining_instances: list) -> None:
        """
        Updates the list of unused DTC instances in the corresponding tmp file.

        :param remaining_instances: updated list to save in tmp file
        """
        with open(SESSION_DIR + "/" + DTC_TMP_FILE, "w") as f:
            json.dump({'list': remaining_instances}, f, default=str)

    @staticmethod
    def remove_cc_instance_from_tmp_file() -> None:
        """
        Clears the customer complaints tmp file.
        """
        with open(SESSION_DIR + "/" + CC_TMP_FILE, 'w') as f:
            # clear list, already used now
            json.dump({'list': []}, f, default=str)

    @staticmethod
    def manual_transition() -> None:
        val = None
        while val != "":
            val = input("\n..............................")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE' state.

        :param userdata: input of state
        :return: outcome of the state ("selected_matching_instance(OBD_CC)" | "no_matching_selected_best_instance" |
                                       "no_instance" | "no_instance_and_CC_already_used")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # load DTC instances from tmp file
        with open(SESSION_DIR + "/" + DTC_TMP_FILE) as f:
            dtc_list = json.load(f)['list']

        customer_complaints_list = []
        try:
            # load customer complaints from tmp file
            with open(SESSION_DIR + "/" + CC_TMP_FILE) as f:
                customer_complaints_list = json.load(f)['list']
        except FileNotFoundError:
            pass

        # case 1: no DTC instance provided, but CC still available
        if len(dtc_list) == 0 and len(customer_complaints_list) == 1:
            # this option leads to the customer complaints being used to generate an artificial DTC instance
            self.remove_cc_instance_from_tmp_file()
            userdata.customer_complaints = customer_complaints_list[0]
            print("no DTCs provided, but customer complaints available..")
            self.manual_transition()
            return "no_instance"

        # case 2: both available
        elif len(dtc_list) > 0 and len(customer_complaints_list) == 1:
            # sub-case 1: matching instance
            for dtc in dtc_list:
                # TODO: check whether DTC matches CC
                match = True
                if match:
                    userdata.selected_instance = dtc
                    dtc_list.remove(dtc)
                    self.remove_dtc_instance_from_tmp_file(dtc_list)
                    print("select matching instance (OBD, CC)..")
                    self.manual_transition()
                    return "selected_matching_instance(OBD_CC)"
            # sub-case 2: no matching instance -> select best instance
            # TODO: select best remaining DTC instance based on some criteria
            userdata.selected_instance = dtc_list[0]
            dtc_list.remove(dtc_list[0])
            self.remove_dtc_instance_from_tmp_file(dtc_list)
            print("DTCs and customer complaints available, but no matching instance..")
            self.manual_transition()
            return "no_matching_selected_best_instance"

        # case 3: no remaining instance and customer complaints already used
        elif len(dtc_list) == 0 and len(customer_complaints_list) == 0:
            print("no more DTC instances and customer complaints already considered..")
            self.manual_transition()
            return "no_instance_and_CC_already_used"

        # case 4: no customer complaints, but remaining DTCs
        else:
            # TODO: select best remaining DTC instance based on some criteria
            selected_dtc = dtc_list[0]
            userdata.selected_instance = selected_dtc
            dtc_list.remove(selected_dtc)
            self.remove_dtc_instance_from_tmp_file(dtc_list)
            print("\nno customer complaints available, selecting DTC instance..")
            print(colored("selected DTC instance: " + selected_dtc, "green", "on_grey", ["bold"]))
            self.manual_transition()
            return "no_matching_selected_best_instance"
