#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
import shutil

import smach
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.data_types.workshop_data import WorkshopData
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class RecVehicleAndProcMetadata(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic receives the vehicle and processes
    the metadata (data about the workshop, mechanic, etc.).
    """

    def __init__(self, data_accessor: DataAccessor, data_provider: DataProvider) -> None:
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self, outcomes=['processed_metadata'], input_keys=[''], output_keys=[''])
        self.data_accessor = data_accessor
        self.data_provider = data_provider

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        print("\n\n############################################")
        print("executing", colored("REC_VEHICLE_AND_PROC_METADATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

    @staticmethod
    def create_session_dir() -> None:
        """
        If not present, it creates the directory for session data.
        If it already exists, it clears the outdated session data.
        """
        if not os.path.exists(SESSION_DIR):
            print(colored("\n------ creating session data directory..", "green", "on_grey", ["bold"]))
            os.makedirs(SESSION_DIR + "/")
        else:
            print(colored("\n------ clearing session data directory..", "green", "on_grey", ["bold"]))
            shutil.rmtree(SESSION_DIR)
            os.makedirs(SESSION_DIR + "/")

    @staticmethod
    def init_classification_log() -> None:
        """
        Initializes the classification log.
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, 'w') as f:
            json.dump([], f, indent=4)

    @staticmethod
    def write_metadata_to_session_dir(workshop_info: WorkshopData) -> None:
        """
        Writes metadata to the session directory.

        :param workshop_info: workshop info (metadata) to be stored in session dir
        """
        with open(SESSION_DIR + '/metadata.json', 'w') as f:
            print(colored("------ writing metadata to session directory..", "green", "on_grey", ["bold"]))
            json.dump(workshop_info.get_json_representation(), f, default=str)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'REC_VEHICLE_AND_PROC_METADATA' state.

        :param userdata: input of the state
        :return: outcome of the state ("processed_metadata")
        """
        self.log_state_info()
        workshop_info = self.data_accessor.get_workshop_info()
        print("max num of parallel recordings:", workshop_info.num_of_parallel_rec)
        print("date:", workshop_info.diag_date)
        self.create_session_dir()
        self.init_classification_log()
        # TODO: save workshop info in KG
        self.write_metadata_to_session_dir(workshop_info)
        self.data_provider.provide_state_transition(StateTransition(
            "REC_VEHICLE_AND_PROC_METADATA", "PROC_CUSTOMER_COMPLAINTS", "processed_metadata"
        ))
        return "processed_metadata"
