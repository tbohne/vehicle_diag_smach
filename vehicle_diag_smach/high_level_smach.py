#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import logging
import os

import smach
import tensorflow as tf
from bs4 import BeautifulSoup
from obd_ontology import knowledge_graph_query_tool
from termcolor import colored

from config import SESSION_DIR, XPS_SESSION_FILE, HISTORICAL_INFO_FILE, CC_TMP_FILE
from vehicle_diag_smach.diagnosis import DiagnosisStateMachine
from high_level_states.rec_vehicle_and_proc_metadata import RecVehicleAndProcMetadata
from high_level_states.proc_customer_complaints import ProcCustomerComplaints
from high_level_states.read_obd_data_and_gen_ontology_instances import ReadOBDDataAndGenOntologyInstances


class EstablishInitialHypothesis(smach.State):
    """
    State in the high-level SMACH that represents situations in which an initial hypothesis is established based
    on the provided information.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_init_hypothesis', 'no_OBD_and_no_CC'],
                             input_keys=['vehicle_specific_instance_data'],
                             output_keys=['hypothesis'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ESTABLISH_INITIAL_HYPOTHESIS' state.

        :param userdata: input of state
        :return: outcome of the state ("established_init_hypothesis" | "no_OBD_and_no_CC")
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

        if len(userdata.vehicle_specific_instance_data['dtc_list']) == 0 and len(initial_hypothesis) == 0:
            # no OBD data + no customer complaints -> insufficient data
            return "no_OBD_and_no_CC"

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

        val = None
        while val != "":
            val = input("\n..............................")

        return "established_init_hypothesis"


class RetrieveHistoricalData(smach.State):
    """
    State in the high-level SMACH that represents situations in which historical information are retrieved for the
    given car (individually, not type), i.e., information that we accumulated in previous repair sessions.
    Optionally, historical data for the car model can be retrieved.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_all_data'],
                             input_keys=['vehicle_specific_instance_data_in'],
                             output_keys=['vehicle_specific_instance_data_out'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'RETRIEVE_HISTORICAL_DATA' state.

        Two kinds of information:
        - historical info for specific vehicle (via VIN)
        - historical info for vehicle type (via model)

        :param userdata: input of state
        :return: outcome of the state ("processed_all_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("RETRIEVE_HISTORICAL_DATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)
        vin = userdata.vehicle_specific_instance_data_in['vin']
        model = userdata.vehicle_specific_instance_data_in['model']

        # TODO: potentially retrieve more historical information (not only DTCs)
        historic_dtcs_by_vin = qt.query_dtcs_by_vin(vin)
        print("DTCs previously recorded in present car:", historic_dtcs_by_vin)
        print("\nmodel to retrieve historical data for:", model, "\n")
        historic_dtcs_by_model = qt.query_dtcs_by_model(model)
        print("DTCs previously recorded in model of present car:", historic_dtcs_by_model)

        with open(SESSION_DIR + "/" + HISTORICAL_INFO_FILE, "w") as f:
            f.write("DTCs previously recorded in car with VIN " + vin + ": " + str(historic_dtcs_by_vin) + "\n")
            f.write("DTCs previously recorded in cars of model " + model + ": " + str(historic_dtcs_by_model) + "\n")

        userdata.vehicle_specific_instance_data_out = userdata.vehicle_specific_instance_data_in

        val = None
        while val != "":
            val = input("\n..............................")

        return "processed_all_data"

class VehicleDiagnosisStateMachine(smach.StateMachine):
    """
    High-level hierarchically structured state machine guiding the entire vehicle diagnosis process.
    """

    def __init__(self):
        super(VehicleDiagnosisStateMachine, self).__init__(
            outcomes=['diag', 'insufficient_data', 'refuted_hypothesis'],
            input_keys=[],
            output_keys=[]
        )
        self.userdata.sm_input = []

        with self:

            self.add('REC_VEHICLE_AND_PROC_METADATA', RecVehicleAndProcMetadata(),
                     transitions={'processed_metadata': 'PROC_CUSTOMER_COMPLAINTS'},
                     remapping={'input': 'sm_input',
                                'user_data': 'sm_input'})

            self.add('PROC_CUSTOMER_COMPLAINTS', ProcCustomerComplaints(),
                     transitions={'received_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES',
                                  'no_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES'},
                     remapping={'user_data': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES', ReadOBDDataAndGenOntologyInstances(),
                     transitions={'processed_OBD_data': 'RETRIEVE_HISTORICAL_DATA',
                                  'no_OBD_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'interview_data': 'sm_input',
                                'vehicle_specific_instance_data': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(),
                     transitions={'established_init_hypothesis': 'DIAGNOSIS',
                                  'no_OBD_and_no_CC': 'insufficient_data'},
                     remapping={'vehicle_specific_instance_data': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_DATA', RetrieveHistoricalData(),
                     transitions={'processed_all_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'vehicle_specific_instance_data_in': 'sm_input',
                                'vehicle_specific_instance_data_out': 'sm_input'})

            self.add('DIAGNOSIS', DiagnosisStateMachine(),
                     transitions={'diag': 'diag',
                                  'refuted_hypothesis': 'refuted_hypothesis'})


def log_info(msg):
    pass


def log_warn(msg):
    pass


def log_debug(msg):
    pass


def log_err(msg):
    print("[ ERROR ] : " + str(msg))


def run():
    """
    Runs the state machine.
    """
    # set custom logging functions
    smach.set_loggers(log_info, log_debug, log_warn, log_err)

    sm = VehicleDiagnosisStateMachine()
    tf.get_logger().setLevel(logging.ERROR)
    outcome = sm.execute()
    # print("OUTCOME:", outcome)


if __name__ == '__main__':
    run()
