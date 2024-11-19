#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import Dict, Union, List

import smach
from obd_ontology import knowledge_graph_query_tool
from obd_ontology import ontology_instance_generator
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OBD_INFO_FILE, CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideInitialHypothesisAndLogContext(smach.State):
    """
    State in the low-level SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self, outcomes=['no_diag'], input_keys=[''], output_keys=['final_output'])
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "yellow", "on_grey", ["bold"]),
              "state..")
        print("############################################")

    @staticmethod
    def read_metadata() -> Dict[str, Union[int, str]]:
        """
        Reads the metadata from the session directory.

        :return: metadata dictionary
        """
        with open(SESSION_DIR + '/metadata.json', 'r') as f:
            return json.load(f)

    @staticmethod
    def read_obd_data() -> Dict[str, Union[str, List[str]]]:
        """
        Reads the OBD data from the session directory.

        :return: OBD data dictionary
        """
        with open(SESSION_DIR + "/" + OBD_INFO_FILE, "r") as f:
            return json.load(f)

    @staticmethod
    def read_classification_ids() -> List[str]:
        """
        Reads the classification IDs from the session directory.

        :return: list of classification IDs
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
        return [classification_entry["Classification ID"] for classification_entry in log_file]

    def read_vehicle_id(self, obd_data) -> str:
        """
        Queries the vehicle ID based on the provided OBD data.

        :param obd_data: OBD data to query vehicle ID for
        :return: vehicle ID
        """
        return self.qt.query_vehicle_instance_by_vin(obd_data["vin"])[0].split("#")[1]

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT' state.

        :param userdata: input of state
        :return: outcome of the state ("no_diag")
        """
        self.log_state_info()
        # TODO: create log file for the failed diagnostic process to improve future diagnosis (missing knowledge etc.)
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "refuted_hypothesis", "no_diag"
        ))
        data = self.read_metadata()
        obd_data = self.read_obd_data()
        classification_ids = self.read_classification_ids()
        vehicle_id = self.read_vehicle_id(obd_data)

        self.instance_gen.extend_knowledge_graph_with_diag_log(
            data["diag_date"], data["max_num_of_parallel_rec"], obd_data["dtc_list"], [], classification_ids, vehicle_id
        )
        userdata.final_output = []
        return "no_diag"
