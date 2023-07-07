#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import knowledge_graph_query_tool
from obd_ontology import ontology_instance_generator
from termcolor import colored

from vehicle_diag_smach.config import OBD_ONTOLOGY_PATH, SESSION_DIR, OBD_INFO_FILE, CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideInitialHypothesisAndLogContext(smach.State):
    """
    State in the low-level SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self, outcomes=['no_diag'], input_keys=[''], output_keys=[''])
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(
            OBD_ONTOLOGY_PATH, local_kb=False, kg_url=kg_url
        )
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False, kg_url=kg_url)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT' state.

        :param userdata: input of state
        :return: outcome of the state ("no_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "yellow", "on_grey", ["bold"]),
              "state..")
        print("############################################")
        # TODO: create log file for the failed diagnostic process to improve future diagnosis (missing knowledge etc.)
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "refuted_hypothesis", "no_diag"
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

        #  read vehicle ID
        vehicle_id = self.qt.query_vehicle_instance_by_vin(obd_data["vin"])[0].split("#")[1]

        # extend KG with `DiagLog` instance
        self.instance_gen.extend_knowledge_graph_with_diag_log(
            data["diag_date"], data["max_num_of_parallel_rec"], obd_data["dtc_list"], [], classification_ids, vehicle_id
        )
        return "no_diag"
