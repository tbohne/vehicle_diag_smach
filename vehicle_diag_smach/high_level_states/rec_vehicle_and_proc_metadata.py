#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
import shutil

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor


class RecVehicleAndProcMetadata(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic receives the vehicle and processes
    the metadata (data about the workshop, mechanic, etc.).
    """

    def __init__(self, data_accessor: DataAccessor):
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        """
        smach.State.__init__(self, outcomes=['processed_metadata'], input_keys=[''], output_keys=[''])
        self.data_accessor = data_accessor

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
        workshop_info = self.data_accessor.get_workshop_info()
        print("max num of parallel recordings:", workshop_info.num_of_parallel_rec)
        print("date:", workshop_info.diag_date)

        # if not present, create directory for session data
        if not os.path.exists(SESSION_DIR):
            print(colored("\n------ creating session data directory..", "green", "on_grey", ["bold"]))
            os.makedirs(SESSION_DIR + "/")
        else:
            # if it already exists, clear outdated session data
            print(colored("\n------ clearing session data directory..", "green", "on_grey", ["bold"]))
            shutil.rmtree(SESSION_DIR)
            os.makedirs(SESSION_DIR + "/")

        # write metadata to session directory
        with open(SESSION_DIR + '/metadata.json', 'w') as f:
            print(colored("------ writing metadata to session directory..", "green", "on_grey", ["bold"]))
            json.dump(workshop_info, f, default=str)

        val = None
        while val != "":
            val = input("\n..............................")

        return "processed_metadata"
