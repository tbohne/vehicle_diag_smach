#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OSCI_SESSION_FILES


class PerformDataManagement(smach.State):
    """
    State in the low-level SMACH that represents situations in which data management is performed, e.g.:
        - upload all the generated session files (data) to the server
        - retrieve the latest trained classification model from the server
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['performed_data_management', 'performed_reduced_data_management'],
                             input_keys=['suggestion_list'],
                             output_keys=['suggestion_list'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PERFORM_DATA_MANAGEMENT' state.

        :param userdata: input of state
        :return: outcome of the state ("performed_data_management" | "performed_reduced_data_management")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PERFORM_DATA_MANAGEMENT", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # TODO: optionally retrieve latest version of trained classifier from server
        print("\nretrieving latest version of trained classifier from server..")

        # TODO: actually read session data
        print("reading customer complaints from session files..")
        print("reading OBD data from session files..")
        print("reading historical info from session files..")
        print("reading user data from session files..")
        print("reading XPS interview data from session files..")

        # determine whether oscillograms have been generated
        osci_session_dir = SESSION_DIR + "/" + OSCI_SESSION_FILES + "/"
        if os.path.exists(osci_session_dir):
            print("reading recorded oscillograms from session files..")
            # TODO:
            #   - EDC (Eclipse Dataspace Connector) communication
            #   - consolidate + upload read session data to server
            print("uploading session data to server..")

            val = None
            while val != "":
                val = input("\n..............................")

            return "performed_data_management"

        # TODO:
        #   - EDC (Eclipse Dataspace Connector) communication
        #   - consolidate + upload read session data to server
        print("uploading reduced session data to server..")
        return "performed_reduced_data_management"
