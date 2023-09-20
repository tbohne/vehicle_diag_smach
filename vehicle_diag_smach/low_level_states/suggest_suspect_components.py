#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List, Dict, Tuple

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

    def __init__(self, data_provider: DataProvider, kg_url: str) -> None:
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
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SUGGEST_SUSPECT_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

    @staticmethod
    def write_components_to_file(suspect_components: List[str]) -> None:
        """
        Writes the suspect components to a session file.

        :param suspect_components: components to be stored in session file
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
            json.dump(suspect_components, f, default=str)

    @staticmethod
    def read_components_from_file() -> List[str]:
        """
        Reads the remaining suspect components from file.

        :return: list of remaining suspect components
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE) as f:
            return json.load(f)

    @staticmethod
    def write_suggestions_to_session_file(selected_instance: str, suspect_components: List[str]) -> None:
        """
        Writes the suggestions to a session file - always the latest ones.

        :param selected_instance: selected DTC instance
        :param suspect_components: list of suggested suspect components
        """
        suggestion = {selected_instance: str(suspect_components)}
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE, 'w') as f:
            json.dump(suggestion, f, default=str)

    def determine_oscilloscope_usage(self, suspect_components: List[str]) -> List[bool]:
        """
        Decides whether an oscilloscope is required / feasible for each component.

        :param suspect_components: components to determine oscilloscope usage for
        :return: booleans representing oscilloscope usage for each component
        """
        oscilloscope_usage = []
        for comp in suspect_components:
            use = self.qt.query_oscilloscope_usage_by_suspect_component(comp)[0]
            print("comp:", comp, "// use oscilloscope:", use)
            oscilloscope_usage.append(use)
        return oscilloscope_usage

    def gen_suggestions(self, selected_instance: str, suspect_components: List[str], oscilloscope_usage: List[bool]) \
            -> Dict[str, Tuple[str, bool]]:
        """
        Generates the suggestion dictionary: {comp: (reason_for, anomaly)}.

        :param selected_instance: selected DTC instance
        :param suspect_components: suggested suspect components
        :param oscilloscope_usage: oscilloscope usage for the suggested components
        :return: suggestion dictionary
        """
        return {
            comp: (
                self.qt.query_diag_association_instance_by_dtc_and_sus_comp(
                    selected_instance, comp
                )[0].split("#")[1], osci
            ) for comp, osci in zip(suspect_components, oscilloscope_usage)
        }

    @staticmethod
    def update_session_file(suspect_components: List[str], suggestions: Dict[str, Tuple[str, bool]]) -> None:
        """
        Everything that is used here should be removed from the tmp file.

        :param suspect_components: suggested components
        :param suggestions: suggestion dictionary
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
            json.dump([c for c in suspect_components if c not in suggestions.keys()], f, default=str)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SUGGEST_SUSPECT_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_suggestions")
        """
        self.log_state_info()
        # should not be queried over and over again - just once for a session
        # -> then suggest as many as possible per execution of the state (write to session files)
        if not os.path.exists(SESSION_DIR + "/" + SUS_COMP_TMP_FILE):
            suspect_components = self.qt.query_suspect_components_by_dtc(userdata.selected_instance)
            ordered_sus_comp = {  # sort suspect components
                int(self.qt.query_priority_id_by_dtc_and_sus_comp(userdata.selected_instance, comp, False)[0]):
                    comp for comp in suspect_components
            }
            suspect_components = [ordered_sus_comp[i] for i in range(len(suspect_components))]
            self.write_components_to_file(suspect_components)
        else:
            suspect_components = self.read_components_from_file()
        print(colored("SUSPECT COMPONENTS: " + str(suspect_components) + "\n", "green", "on_grey", ["bold"]))
        self.write_suggestions_to_session_file(userdata.selected_instance, suspect_components)
        oscilloscope_usage = self.determine_oscilloscope_usage(suspect_components)
        suggestions = self.gen_suggestions(userdata.selected_instance, suspect_components, oscilloscope_usage)
        userdata.suggestion_list = suggestions
        self.update_session_file(suspect_components, suggestions)

        if True in oscilloscope_usage:
            print("\n--> there is at least one suspect component that can be diagnosed using an oscilloscope..")
            self.data_provider.provide_state_transition(StateTransition(
                "SUGGEST_SUSPECT_COMPONENTS", "CLASSIFY_COMPONENTS", "provided_suggestions"
            ))
            return "provided_suggestions"
        print("none of the identified suspect components can be diagnosed with an oscilloscope..")
