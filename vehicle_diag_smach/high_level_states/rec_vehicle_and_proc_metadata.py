#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
import shutil
from datetime import date

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR


class RecVehicleAndProcMetadata(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic receives the vehicle and processes
    the metadata (data about the workshop, mechanic, etc.).
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['processed_metadata'], input_keys=[''], output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'REC_VEHICLE_AND_PROC_METADATA' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_metadata")
        """
        print("\n\n############################################")
        print("executing", colored("REC_VEHICLE_AND_PROC_METADATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")
        print()
        # TODO: read from updated GUI
        # GUI.run_gui()
        user_data = {
            "workshop_name": "workshop_one",
            "zipcode": "12345",
            "workshop_id": "00000",
            "mechanic_id": "99999",
            # how many parallel measurements are possible at most (based on workshop equipment / oscilloscope channels)
            "max_number_of_parallel_recordings": "4",
            "date": date.today()
        }

        for k in user_data.keys():
            print(k + ": " + str(user_data[k]))

        # if not present, create directory for session data
        if not os.path.exists(SESSION_DIR):
            print(colored("\n------ creating session data directory..", "green", "on_grey", ["bold"]))
            os.makedirs(SESSION_DIR + "/")
        else:
            # if it already exists, clear outdated session data
            print(colored("\n------ clearing session data directory..", "green", "on_grey", ["bold"]))
            shutil.rmtree(SESSION_DIR)
            os.makedirs(SESSION_DIR + "/")

        # write user data to session directory
        with open(SESSION_DIR + '/user_data.json', 'w') as f:
            print(colored("------ writing user data to session directory..", "green", "on_grey", ["bold"]))
            json.dump(user_data, f, default=str)

        val = None
        while val != "":
            val = input("\n..............................")

        return "processed_metadata"
