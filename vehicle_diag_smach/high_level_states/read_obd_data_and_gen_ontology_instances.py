#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import ontology_instance_generator
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OBD_INFO_FILE, DTC_TMP_FILE, OBD_ONTOLOGY_PATH, KG_URL
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider


class ReadOBDDataAndGenOntologyInstances(smach.State):
    """
    State in the high-level SMACH that represents situations in which the OBD information are read from the ECU.
    Based on the read information, ontology instances are generated, i.e., the vehicle-specific instance data
    is entered into the knowledge graph.
    """

    def __init__(self, data_accessor: DataAccessor, data_provider: DataProvider):
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['processed_OBD_data', 'no_DTC_data'],
                             input_keys=[''],
                             output_keys=['vehicle_specific_instance_data'])
        self.data_accessor = data_accessor
        self.data_provider = data_provider

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_OBD_data" | "no_DTC_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "yellow", "on_grey", ["bold"]),
              "state..")
        print("############################################")
        obd_data = self.data_accessor.get_obd_data()

        if len(obd_data.dtc_list) == 0:
            self.data_provider.provide_state_transition(StateTransition(
                "READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "ESTABLISH_INITIAL_HYPOTHESIS", "no_DTC_data"
            ))
            userdata.vehicle_specific_instance_data = obd_data
            return "no_DTC_data"

        # write OBD data to session file
        with open(SESSION_DIR + "/" + OBD_INFO_FILE, "w") as f:
            json.dump(obd_data.get_json_representation(), f, default=str)
        # also create tmp file for unused DTC instances
        with open(SESSION_DIR + "/" + DTC_TMP_FILE, "w") as f:
            dtc_tmp = {'list': obd_data.dtc_list}
            json.dump(dtc_tmp, f, default=str)

        # read workshop metadata
        with open(SESSION_DIR + "/" + "metadata.json") as f:
            workshop_info = json.load(f)
            max_num_of_parallel_rec = int(workshop_info["max_num_of_parallel_rec"])
            diag_date = workshop_info["diag_date"]

        # extend knowledge graph with read OBD data (if the vehicle instance already exists, it will be extended)
        instance_gen = ontology_instance_generator.OntologyInstanceGenerator(
            OBD_ONTOLOGY_PATH, local_kb=False, kg_url=KG_URL
        )
        for dtc in obd_data.dtc_list:
            instance_gen.extend_knowledge_graph_with_vehicle_data(
                obd_data.model, obd_data.hsn, obd_data.tsn, obd_data.vin, dtc
            )
        userdata.vehicle_specific_instance_data = obd_data
        self.data_provider.provide_state_transition(StateTransition(
            "READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "RETRIEVE_HISTORICAL_DATA", "processed_OBD_data"
        ))
        return "processed_OBD_data"
