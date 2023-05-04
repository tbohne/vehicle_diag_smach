#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import logging

import smach
import tensorflow as tf

from vehicle_diag_smach.diagnosis import DiagnosisStateMachine
from vehicle_diag_smach.high_level_states.establish_initial_hypothesis import EstablishInitialHypothesis
from vehicle_diag_smach.high_level_states.proc_customer_complaints import ProcCustomerComplaints
from vehicle_diag_smach.high_level_states.read_obd_data_and_gen_ontology_instances import \
    ReadOBDDataAndGenOntologyInstances
from vehicle_diag_smach.high_level_states.rec_vehicle_and_proc_metadata import RecVehicleAndProcMetadata
from vehicle_diag_smach.high_level_states.retrieve_historical_data import RetrieveHistoricalData
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.io.local_data_accessor import LocalDataAccessor


class VehicleDiagnosisStateMachine(smach.StateMachine):
    """
    High-level hierarchically structured state machine guiding the entire vehicle diagnosis process.
    """

    def __init__(self, data_accessor: DataAccessor):
        super(VehicleDiagnosisStateMachine, self).__init__(
            outcomes=['diag', 'insufficient_data', 'refuted_hypothesis'],
            input_keys=[],
            output_keys=[]
        )
        self.data_accessor = data_accessor
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

            self.add('READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES', ReadOBDDataAndGenOntologyInstances(self.data_accessor),
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
    print("[ INFO ] : " + str(msg))


def log_warn(msg):
    print("[ WARNING ] : " + str(msg))


def log_debug(msg):
    print("[ DEBUG ] : " + str(msg))


def log_err(msg):
    print("[ ERROR ] : " + str(msg))


if __name__ == '__main__':
    # set custom logging functions
    smach.set_loggers(log_info, log_debug, log_warn, log_err)

    # init local implementations of I/O interfaces
    data_acc = LocalDataAccessor()

    sm = VehicleDiagnosisStateMachine(data_acc)
    tf.get_logger().setLevel(logging.ERROR)
    sm.execute()
