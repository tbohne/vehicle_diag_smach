#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import Dict, List, Union

import smach
from obd_ontology import ontology_instance_generator, knowledge_graph_query_tool
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OBD_INFO_FILE, CLASSIFICATION_LOG_FILE, SUGGESTION_SESSION_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideDiagAndShowTrace(smach.State):
    """
    State in the low-level SMACH that represents situations in which the diagnosis is provided in combination with
    a detailed trace of all the relevant information that led to it (explanatory report). Additionally, the diagnosis
    is entered into the knowledge graph.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self, outcomes=['uploaded_diag'], input_keys=['diagnosis'], output_keys=['final_output'])
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
        print("executing", colored("PROVIDE_DIAG_AND_SHOW_TRACE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

    @staticmethod
    def construct_fault_paths(diagnosis: Dict[str, List[List[str]]], anomalous_comp: str) -> List[str]:
        """
        Constructs the fault paths from the given diagnosis.

        :param diagnosis: diagnosis to construct fault paths from
        :param anomalous_comp: entry component
        :return: constructed fault paths
        """
        paths = [diagnosis[anomalous_comp][branch][::-1] for branch in range(len(diagnosis[anomalous_comp]))]
        return ["".join([path[i] if i == len(path) - 1 else path[i] + " -> " for i in range(len(path))]) for path in
                paths]

    def retrieve_fault_condition_id(self) -> str:
        """
        Retrieves the fault condition ID for the fault path. Reads the DTC suggestion under the assumption that it is
        always the latest one.

        :return: fault condition ID
        """
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
            suggestions = json.load(f)
        assert len(suggestions.keys()) == 1
        dtc = list(suggestions.keys())[0]
        return self.qt.query_fault_condition_instance_by_code(dtc)[0].split("#")[1]

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

    def read_vehicle_id(self, obd_data: Dict[str, Union[str, List[str]]]) -> str:
        """
        Queries the vehicle ID based on the provided OBD data.

        :param obd_data: OBD data to query vehicle ID for
        :return: vehicle ID
        """
        return self.qt.query_vehicle_instance_by_vin(obd_data["vin"])[0].split("#")[1]

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        Enters the diagnosis into the KG. It is important to log the whole context - everything that could be meaningful
        in the long run, this is where we collect the data that we initially lacked, e.g., for automated data-driven
        RCA.

        :param userdata: input of state
        :return: outcome of the state ("uploaded_diag")
        """
        self.log_state_info()
        fault_paths = {}

        for key in userdata.diagnosis.keys():
            print("\nidentified anomalous component:", key)
            branching_paths = self.construct_fault_paths(userdata.diagnosis, key)
            fault_condition_id = self.retrieve_fault_condition_id()
            for branching_path in branching_paths:
                fault_path_id = self.instance_gen.extend_knowledge_graph_with_fault_path(
                    branching_path, fault_condition_id
                )
                fault_paths[fault_path_id] = branching_path

        self.data_provider.provide_diagnosis(list(fault_paths.values()))
        userdata.final_output = list(fault_paths.values())
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_DIAG_AND_SHOW_TRACE", "diag", "uploaded_diag"
        ))

        data = self.read_metadata()
        obd_data = self.read_obd_data()
        classification_ids = self.read_classification_ids()
        vehicle_id = self.read_vehicle_id(obd_data)

        self.instance_gen.extend_knowledge_graph_with_diag_log(
            data["diag_date"], data["max_num_of_parallel_rec"], obd_data["dtc_list"], list(fault_paths.keys()),
            classification_ids, vehicle_id
        )
        return "uploaded_diag"
