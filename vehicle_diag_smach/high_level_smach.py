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
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor
from vehicle_diag_smach.io.local_data_accessor import LocalDataAccessor
from vehicle_diag_smach.io.local_data_provider import LocalDataProvider
from vehicle_diag_smach.io.local_model_accessor import LocalModelAccessor


class VehicleDiagnosisStateMachine(smach.StateMachine):
    """
    High-level hierarchically structured state machine guiding the entire vehicle diagnosis process.
    """

    def __init__(self, data_accessor: DataAccessor, model_accessor: ModelAccessor, data_provider: DataProvider):
        """
        Initializes the high-level state machine.

        :param data_accessor: implementation of the data accessor interface
        :param model_accessor: implementation of the model accessor interface
        :param data_provider: implementation of the data provider interface
        """
        super(VehicleDiagnosisStateMachine, self).__init__(
            outcomes=['diag', 'insufficient_data', 'refuted_hypothesis'],
            input_keys=[],
            output_keys=[]
        )
        self.data_accessor = data_accessor
        self.model_accessor = model_accessor
        self.data_provider = data_provider
        self.userdata.sm_input = []

        with self:
            self.add('REC_VEHICLE_AND_PROC_METADATA', RecVehicleAndProcMetadata(self.data_accessor),
                     transitions={'processed_metadata': 'PROC_CUSTOMER_COMPLAINTS'},
                     remapping={'input': 'sm_input',
                                'user_data': 'sm_input'})

            self.add('PROC_CUSTOMER_COMPLAINTS', ProcCustomerComplaints(self.data_accessor, self.data_provider),
                     transitions={'received_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES',
                                  'no_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES'},
                     remapping={'user_data': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES', ReadOBDDataAndGenOntologyInstances(self.data_accessor),
                     transitions={'processed_OBD_data': 'RETRIEVE_HISTORICAL_DATA',
                                  'no_OBD_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'interview_data': 'sm_input',
                                'vehicle_specific_instance_data': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(self.data_provider),
                     transitions={'established_init_hypothesis': 'DIAGNOSIS',
                                  'no_OBD_and_no_CC': 'insufficient_data'},
                     remapping={'vehicle_specific_instance_data': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_DATA', RetrieveHistoricalData(),
                     transitions={'processed_all_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'vehicle_specific_instance_data_in': 'sm_input',
                                'vehicle_specific_instance_data_out': 'sm_input'})

            self.add('DIAGNOSIS', DiagnosisStateMachine(self.model_accessor, self.data_accessor, self.data_provider),
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


if __name__ == '__main__':
    # set custom logging functions
    smach.set_loggers(log_info, log_debug, log_warn, log_err)

    # init local implementations of I/O interfaces
    data_acc = LocalDataAccessor()
    model_acc = LocalModelAccessor()
    data_prov = LocalDataProvider()

    sm = VehicleDiagnosisStateMachine(data_acc, model_acc, data_prov)
    tf.get_logger().setLevel(logging.ERROR)
    sm.execute()
