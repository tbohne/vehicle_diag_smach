#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from bs4 import BeautifulSoup
from obd_ontology import ontology_instance_generator, knowledge_graph_query_tool
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, XPS_SESSION_FILE, HISTORICAL_INFO_FILE, CC_TMP_FILE, \
    OBD_ONTOLOGY_PATH, KG_URL, OBD_INFO_FILE, CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class EstablishInitialHypothesis(smach.State):
    """
    State in the high-level SMACH that represents situations in which an initial hypothesis is established based
    on the provided information.
    """

    def __init__(self, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['established_init_hypothesis', 'no_DTC_and_no_CC'],
                             input_keys=['vehicle_specific_instance_data'],
                             output_keys=['hypothesis'])
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(
            OBD_ONTOLOGY_PATH, local_kb=False, kg_url=KG_URL
        )
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False, kg_url=KG_URL)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ESTABLISH_INITIAL_HYPOTHESIS' state.

        :param userdata: input of state
        :return: outcome of the state ("established_init_hypothesis" | "no_DTC_and_no_CC")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("ESTABLISH_INITIAL_HYPOTHESIS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        print("\nreading customer complaints session protocol..")
        initial_hypothesis = ""
        try:
            with open(SESSION_DIR + "/" + XPS_SESSION_FILE) as f:
                data = f.read()
                session_data = BeautifulSoup(data, 'xml')
                for tag in session_data.find_all('rating', {'type': 'heuristic'}):
                    initial_hypothesis = tag.parent['objectName']
        except FileNotFoundError:
            print("no customer complaints available..")

        if len(userdata.vehicle_specific_instance_data.dtc_list) == 0 and len(initial_hypothesis) == 0:
            # no OBD data + no customer complaints -> insufficient data
            self.data_provider.provide_state_transition(StateTransition(
                "ESTABLISH_INITIAL_HYPOTHESIS", "insufficient_data", "no_DTC_and_no_CC"
            ))

            # TODO: generate `DiagLog` instance

            # read meta data
            with open(SESSION_DIR + '/metadata.json', 'r') as f:
                data = json.load(f)
            # read OBD data
            with open(SESSION_DIR + "/" + OBD_INFO_FILE, "r") as f:
                obd_data = json.load(f)

            # read vehicle ID
            vehicle_id = self.qt.query_vehicle_instance_by_vin(obd_data["vin"])[0].split("#")[1]

            # extend KG with `DiagLog` instance
            self.instance_gen.extend_knowledge_graph_with_diag_log(
                data["diag_date"], data["max_num_of_parallel_rec"], obd_data["dtc_list"], [], [], vehicle_id
            )
            return "no_DTC_and_no_CC"

        print("reading historical information..")
        with open(SESSION_DIR + "/" + HISTORICAL_INFO_FILE) as f:
            data = f.read()

        if len(initial_hypothesis) > 0:
            print("initial hypothesis based on customer complaints available..")
            print("initial hypothesis:", initial_hypothesis)
            userdata.hypothesis = initial_hypothesis
            unused_cc = {'list': [initial_hypothesis]}
            with open(SESSION_DIR + "/" + CC_TMP_FILE, 'w') as f:
                json.dump(unused_cc, f, default=str)
        else:
            print("no initial hypothesis based on customer complaints..")

        # TODO: use historical data to refine initial hypothesis (e.g. to deny certain hypotheses)
        print("establish hypothesis..")
        self.data_provider.provide_state_transition(StateTransition(
            "ESTABLISH_INITIAL_HYPOTHESIS", "DIAGNOSIS", "established_init_hypothesis"
        ))
        return "established_init_hypothesis"
