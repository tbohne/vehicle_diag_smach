#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import knowledge_graph_query_tool
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, SUS_COMP_TMP_FILE, SUGGESTION_SESSION_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class SuggestSuspectComponents(smach.State):
    """
    State in the low-level SMACH that represents situations in which suspect components (physical components) in the
    vehicle are suggested to be investigated based on the available information (OBD, CC, etc.).
    """

    def __init__(self, data_provider: DataProvider, kg_url: str):
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['provided_suggestions'],
                             input_keys=['selected_instance', 'generated_instance'],
                             output_keys=['suggestion_list'])
        self.data_provider = data_provider
        self.kg_url = kg_url

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SUGGEST_SUSPECT_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_suggestions")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SUGGEST_SUSPECT_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

        # print("generated instance:", userdata.generated_instance)
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False, kg_url=self.kg_url)

        # should not be queried over and over again - just once for a session
        # -> then suggest as many as possible per execution of the state (write to session files)
        if not os.path.exists(SESSION_DIR + "/" + SUS_COMP_TMP_FILE):
            suspect_components = qt.query_suspect_components_by_dtc(userdata.selected_instance)
            # sort suspect components
            ordered_sus_comp = {
                int(qt.query_priority_id_by_dtc_and_sus_comp(userdata.selected_instance, comp, False)[0]):
                    comp for comp in suspect_components
            }
            suspect_components = [ordered_sus_comp[i] for i in range(len(suspect_components))]

            # write suspect components to session file
            with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
                json.dump(suspect_components, f, default=str)
        else:
            # read remaining suspect components from file
            with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE) as f:
                suspect_components = json.load(f)

        print(colored("SUSPECT COMPONENTS: " + str(suspect_components) + "\n", "green", "on_grey", ["bold"]))

        # write suggestions to session file - always the latest ones
        suggestion = {userdata.selected_instance: str(suspect_components)}
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE, 'w') as f:
            json.dump(suggestion, f, default=str)

        # decide whether oscilloscope required
        oscilloscope_usage = []
        for comp in suspect_components:
            use = qt.query_oscilloscope_usage_by_suspect_component(comp)[0]
            print("comp:", comp, "// use oscilloscope:", use)
            oscilloscope_usage.append(use)

        suggestion_list = {
            comp: (
                qt.query_diag_association_instance_by_dtc_and_sus_comp(
                    userdata.selected_instance, comp
                )[0].split("#")[1], osci
            ) for comp, osci in zip(suspect_components, oscilloscope_usage)
        }
        userdata.suggestion_list = suggestion_list

        # everything that is used here should be removed from the tmp file
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
            json.dump([c for c in suspect_components if c not in suggestion_list.keys()], f, default=str)

        if True in oscilloscope_usage:
            print("\n--> there is at least one suspect component that can be diagnosed using an oscilloscope..")
            self.data_provider.provide_state_transition(StateTransition(
                "SUGGEST_SUSPECT_COMPONENTS", "CLASSIFY_COMPONENTS", "provided_suggestions"
            ))
            return "provided_suggestions"
        print("none of the identified suspect components can be diagnosed with an oscilloscope..")
