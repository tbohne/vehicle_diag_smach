#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import ontology_instance_generator, knowledge_graph_query_tool
from termcolor import colored

from vehicle_diag_smach.config import OBD_ONTOLOGY_PATH, SESSION_DIR, OBD_INFO_FILE, CLASSIFICATION_LOG_FILE, \
    SUGGESTION_SESSION_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideDiagAndShowTrace(smach.State):
    """
    State in the low-level SMACH that represents situations in which the diagnosis is provided in combination with
    a detailed trace of all the relevant information that lead to it. Additionally, the diagnosis is entered into the
    knowledge graph.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['uploaded_diag'],
                             input_keys=['diagnosis'],
                             output_keys=[''])
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        :param userdata: input of state
        :return: outcome of the state ("uploaded_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_DIAG_AND_SHOW_TRACE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # TODO: enter diagnosis into KG
        #   - it's important to log the whole context - everything that could be meaningful
        #   - in the long run, this is where we collect the data that we initially lacked, e.g., for automated
        #     data-driven RCA
        #   - to be logged (diagnosis together with):
        #       - associated symptoms, DTCs, components (distinguishing root causes and side effects) etc.
        # TODO: show diagnosis + trace

        fault_paths = {}
        for key in userdata.diagnosis.keys():
            print("\nidentified anomalous component:", key)
            path = userdata.diagnosis[key][::-1]
            path = [path[i] if i == len(path) - 1 else path[i] + " -> " for i in range(len(path))]
            fault_path = "".join(path)

            # find out fault condition ID for this fault path
            # read DTC suggestion - assumption: it is always the latest suggestion
            with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
                suggestions = json.load(f)
            assert len(suggestions.keys()) == 1
            dtc = list(suggestions.keys())[0]
            fault_condition_id = self.qt.query_fault_condition_instance_by_code(dtc)[0].split("#")[1]
            fault_path_id = self.instance_gen.extend_knowledge_graph_with_fault_path(fault_path, fault_condition_id)
            fault_paths[fault_path_id] = fault_path

        self.data_provider.provide_diagnosis(list(fault_paths.values()))
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_DIAG_AND_SHOW_TRACE", "diag", "uploaded_diag"
        ))

        # read meta data
        with open(SESSION_DIR + '/metadata.json', 'r') as f:
            data = json.load(f)
        # read OBD data
        with open(SESSION_DIR + "/" + OBD_INFO_FILE, "r") as f:
            obd_data = json.load(f)

        # read classification IDs
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
        classification_ids = [classification_entry["Classification ID"] for classification_entry in log_file]

        # read vehicle ID
        vehicle_id = self.qt.query_vehicle_instance_by_vin(obd_data["vin"])[0].split("#")[1]

        # extend KG with `DiagLog` instance
        self.instance_gen.extend_knowledge_graph_with_diag_log(
            data["diag_date"], data["max_num_of_parallel_rec"], obd_data["dtc_list"], list(fault_paths.keys()),
            classification_ids, vehicle_id
        )
        return "uploaded_diag"
