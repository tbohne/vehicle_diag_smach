#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, DTC_TMP_FILE, CC_TMP_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class SelectBestUnusedErrorCodeInstance(smach.State):
    """
    State in the low-level SMACH that represents situations in which a best-suited, unused DTC instance is
    selected for further processing.
    """

    def __init__(self, data_provider: DataProvider) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['selected_matching_instance(OBD_CC)', 'no_matching_selected_best_instance',
                                       'no_instance', 'no_instance_and_CC_already_used'],
                             input_keys=[''],
                             output_keys=['selected_instance', 'customer_complaints'])
        self.data_provider = data_provider

    @staticmethod
    def remove_dtc_instance_from_tmp_file(remaining_instances: List[str]) -> None:
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
            json.dump({'list': []}, f, default=str)  # clear list, already used now

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

    @staticmethod
    def load_dtc_instances() -> List[str]:
        """
        Loads the DTC instances from the tmp file.

        :return: list of DTCs
        """
        with open(SESSION_DIR + "/" + DTC_TMP_FILE) as f:
            return json.load(f)['list']

    @staticmethod
    def load_customer_complaints() -> List[str]:
        """
        Loads the customer complaints from the session file.

        :return: list of customer complaints
        """
        customer_complaints_list = []
        try:
            with open(SESSION_DIR + "/" + CC_TMP_FILE) as f:
                customer_complaints_list = json.load(f)['list']
        except FileNotFoundError:
            pass
        return customer_complaints_list

    def no_dtc_but_customer_complaints(self) -> None:
        """
        This option leads to the customer complaints being used to generate an artificial DTC instance.
        """
        self.remove_cc_instance_from_tmp_file()
        print("no DTCs provided, but customer complaints available..")
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC", "no_instance"
        ))

    def dtc_match(self, dtc_list: List[str], dtc: str) -> None:
        """
        Handles the case of a matching DTC instance.

        :param dtc_list: list of DTCs
        :param dtc: matching DTC
        """
        dtc_list.remove(dtc)
        self.remove_dtc_instance_from_tmp_file(dtc_list)
        print("select matching instance (OBD, CC)..")
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "SUGGEST_SUSPECT_COMPONENTS", "selected_matching_instance(OBD_CC)"
        ))

    def no_matching_instance(self, dtc_list: List[str]) -> None:
        """
        Handles 'no match' cases.

        :param dtc_list: list of DTCs
        """
        dtc_list.remove(dtc_list[0])
        self.remove_dtc_instance_from_tmp_file(dtc_list)
        print("DTCs and customer complaints available, but no matching instance..")
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "SUGGEST_SUSPECT_COMPONENTS", "no_matching_selected_best_instance"
        ))

    def no_remaining_instances_and_customer_complaints_used(self) -> None:
        """
        Handles cases with no remaining DTC instances and already used customer complaints.
        """
        print("no more DTC instances and customer complaints already considered..")
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "NO_PROBLEM_DETECTED_CHECK_SENSOR",
            "no_instance_and_CC_already_used"
        ))

    def no_customer_complaints_but_remaining_dtcs(self, dtc_list: List[str], selected_dtc: str) -> None:
        """
        Handles cases with no customer complaints, but remaining DTCs.

        :param dtc_list: list of DTCs
        :param selected_dtc: selected DTC
        """
        dtc_list.remove(selected_dtc)
        self.remove_dtc_instance_from_tmp_file(dtc_list)
        print("\nno customer complaints available, selecting DTC instance..")
        print(colored("selected DTC instance: " + selected_dtc, "green", "on_grey", ["bold"]))
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "SUGGEST_SUSPECT_COMPONENTS",
            "no_matching_selected_best_instance"
        ))

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE' state.

        :param userdata: input of state
        :return: outcome of the state ("selected_matching_instance(OBD_CC)" | "no_matching_selected_best_instance" |
                                       "no_instance" | "no_instance_and_CC_already_used")
        """
        self.log_state_info()
        dtc_list = self.load_dtc_instances()
        customer_complaints_list = self.load_customer_complaints()

        # case 1: no DTC instance provided, but CC still available
        if len(dtc_list) == 0 and len(customer_complaints_list) == 1:
            self.no_dtc_but_customer_complaints()
            userdata.customer_complaints = customer_complaints_list[0]
            return "no_instance"
        # case 2: both available
        elif len(dtc_list) > 0 and len(customer_complaints_list) == 1:
            for dtc in dtc_list:
                match = True  # TODO: check whether DTC matches CC
                if match:  # sub-case 1: matching instance
                    self.dtc_match(dtc_list, dtc)
                    userdata.selected_instance = dtc
                    return "selected_matching_instance(OBD_CC)"
            # sub-case 2: no matching instance -> select best instance
            # TODO: select best remaining DTC instance based on some criteria
            userdata.selected_instance = dtc_list[0]
            self.no_matching_instance(dtc_list)
            return "no_matching_selected_best_instance"
        # case 3: no remaining instance and customer complaints already used
        elif len(dtc_list) == 0 and len(customer_complaints_list) == 0:
            self.no_remaining_instances_and_customer_complaints_used()
            return "no_instance_and_CC_already_used"
        else:  # case 4: no customer complaints, but remaining DTCs
            # TODO: select best remaining DTC instance based on some criteria
            selected_dtc = dtc_list[0]
            userdata.selected_instance = selected_dtc
            self.no_customer_complaints_but_remaining_dtcs(dtc_list, selected_dtc)
            return "no_matching_selected_best_instance"
