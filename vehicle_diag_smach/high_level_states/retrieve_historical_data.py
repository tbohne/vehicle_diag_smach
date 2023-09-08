#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os
from typing import List

import smach
from obd_ontology import knowledge_graph_query_tool
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, HISTORICAL_INFO_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class RetrieveHistoricalData(smach.State):
    """
    State in the high-level SMACH that represents situations in which historical information are retrieved for the
    given car (individually, not type), i.e., information that we accumulated in previous repair sessions.
    Optionally, historical data for the car model can be retrieved.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['processed_all_data'],
                             input_keys=['vehicle_specific_instance_data_in'],
                             output_keys=['vehicle_specific_instance_data_out'])
        self.data_provider = data_provider
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("RETRIEVE_HISTORICAL_DATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

    @staticmethod
    def write_historical_info_to_session_dir(vin: str, historic_dtcs_by_vin: List[str], model: str,
                                             historic_dtcs_by_model: List[str]) -> None:
        """
        Writes the historical information to the session directory.

        :param vin: vehicle identification number of the considered car
        :param historic_dtcs_by_vin: DTCs previously recorded in the considered car
        :param model: model of the considered car
        :param historic_dtcs_by_model: DTCs previously recorded in instances of the considered model
        """
        with open(SESSION_DIR + "/" + HISTORICAL_INFO_FILE, "w") as f:
            f.write("DTCs previously recorded in car with VIN " + vin + ": " + str(historic_dtcs_by_vin) + "\n")
            f.write("DTCs previously recorded in cars of model " + model + ": " + str(historic_dtcs_by_model) + "\n")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'RETRIEVE_HISTORICAL_DATA' state.

        Two kinds of information:
        - historical info for specific vehicle (via VIN)
        - historical info for vehicle type (via model)

        :param userdata: input of the state
        :return: outcome of the state ("processed_all_data")
        """
        self.log_state_info()
        vin = userdata.vehicle_specific_instance_data_in.vin
        model = userdata.vehicle_specific_instance_data_in.model

        # TODO: potentially retrieve more historical information (not only DTCs)
        historic_dtcs_by_vin = self.qt.query_dtcs_by_vin(vin)
        print("DTCs previously recorded in present car:", historic_dtcs_by_vin)
        print("\nmodel to retrieve historical data for:", model, "\n")
        historic_dtcs_by_model = self.qt.query_dtcs_by_model(model)
        print("DTCs previously recorded in model of present car:", historic_dtcs_by_model)

        self.write_historical_info_to_session_dir(vin, historic_dtcs_by_vin, model, historic_dtcs_by_model)
        userdata.vehicle_specific_instance_data_out = userdata.vehicle_specific_instance_data_in
        self.data_provider.provide_state_transition(StateTransition(
            "RETRIEVE_HISTORICAL_DATA", "ESTABLISH_INITIAL_HYPOTHESIS", "processed_all_data"
        ))
        return "processed_all_data"
