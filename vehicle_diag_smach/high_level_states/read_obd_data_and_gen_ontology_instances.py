#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import smach
from obd_ontology import ontology_instance_generator
from termcolor import colored

from vehicle_diag_smach.config import SAMPLE_OBD_LOG, SESSION_DIR, OBD_INFO_FILE, DTC_TMP_FILE, OBD_ONTOLOGY_PATH


class ReadOBDDataAndGenOntologyInstances(smach.State):
    """
    State in the high-level SMACH that represents situations in which the OBD information are read from the ECU.
    Based on the read information, an ontology instance is generated, i.e., the vehicle-specific instance data
    is entered into the knowledge graph.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['processed_OBD_data', 'no_OBD_data'],
                             input_keys=[''],
                             output_keys=['vehicle_specific_instance_data'])

    @staticmethod
    def parse_obd_logfile() -> dict:
        print("\nprocessing OBD log file..")
        obd_data = {"dtc_list": [], "model": "", "hsn": "", "tsn": "", "vin": ""}

        with open(SAMPLE_OBD_LOG) as f:
            obd_lines = f.readlines()

        # TODO: parse DTCs from OBD log (above)
        obd_data['dtc_list'] = ['P2563']
        obd_data['model'] = "Mazda 3"
        obd_data['hsn'] = "849357984"
        obd_data['tsn'] = "453948539"
        obd_data['vin'] = "1234567890ABCDEFGHJKLMNPRSTUVWXYZ"

        for k in obd_data.keys():
            print(k + ": " + str(obd_data[k]))
        print()

        return obd_data

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_OBD_data" | "no_OBD_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "yellow", "on_grey", ["bold"]),
              "state..")
        print("############################################")
        vehicle_specific_instance_data = self.parse_obd_logfile()
        if len(vehicle_specific_instance_data['dtc_list']) == 0:
            return "no_OBD_data"

        # write OBD data to session file
        with open(SESSION_DIR + "/" + OBD_INFO_FILE, "w") as f:
            json.dump(vehicle_specific_instance_data, f, default=str)
        # also create tmp file for unused DTC instances
        with open(SESSION_DIR + "/" + DTC_TMP_FILE, "w") as f:
            dtc_tmp = {'list': vehicle_specific_instance_data['dtc_list']}
            json.dump(dtc_tmp, f, default=str)

        # extend knowledge graph with read OBD data (if the vehicle instance already exists, it will be extended)
        instance_gen = ontology_instance_generator.OntologyInstanceGenerator(OBD_ONTOLOGY_PATH, local_kb=False)
        for dtc in vehicle_specific_instance_data['dtc_list']:
            instance_gen.extend_knowledge_graph(
                vehicle_specific_instance_data['model'], vehicle_specific_instance_data['hsn'],
                vehicle_specific_instance_data['tsn'], vehicle_specific_instance_data['vin'], dtc
            )

        val = None
        while val != "":
            val = input("\n..............................")

        userdata.vehicle_specific_instance_data = vehicle_specific_instance_data
        return "processed_OBD_data"
