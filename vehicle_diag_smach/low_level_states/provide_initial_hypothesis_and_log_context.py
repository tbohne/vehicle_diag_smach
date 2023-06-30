#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import knowledge_graph_query_tool
from obd_ontology import ontology_instance_generator
from termcolor import colored

from vehicle_diag_smach.config import OBD_ONTOLOGY_PATH, SESSION_DIR, OBD_INFO_FILE, KG_URL
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ProvideInitialHypothesisAndLogContext(smach.State):
    """
    State in the low-level SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self, outcomes=['no_diag'], input_keys=[''], output_keys=[''])
        self.data_provider = data_provider

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

        # TODO: generate `DiagLog` instance
        instance_gen = ontology_instance_generator.OntologyInstanceGenerator(
            OBD_ONTOLOGY_PATH, local_kb=False, kg_url=KG_URL
        )

        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False, kg_url=KG_URL)

        with open(SESSION_DIR + '/metadata.json', 'r') as f:
            data = json.load(f)

        with open(SESSION_DIR + "/" + OBD_INFO_FILE, "r") as f:
            obd_data = json.load(f)

        dtc_instances = [qt.query_dtc_instance_by_code(dtc) for dtc in obd_data["dtc_list"]]

        # TODO: extend ontology instance generation
        # instance_gen.extend_knowledge_graph_with_diag_log(
        #     data["diag_date"], data["max_num_of_parallel_rec"], dtc_instances, [],
        # )

        return "no_diag"
