#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import subprocess
import sys
import time
from os import path

import smach
from bs4 import BeautifulSoup

import config

import rospy
import smach_ros

sys.path.append(path.abspath('../AW_40_GUI'))
sys.path.append(path.abspath('../OBDOntology'))

from GUI import run_gui
from ontology_query_tool import OntologyQueryTool


class ProcUserInfo(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['read_user_info'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing PROC_USER_INFO state..")
        print("############################################")
        time.sleep(10)
        return "read_user_info"


class RecCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['complaints_received', 'no_complaints'],
                             input_keys=['input_info'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        print("launching customer XPS..")
        subprocess.call(['java', '-jar', config.CUSTOMER_XPS])

    def execute(self, userdata):
        print("############################################")
        print("executing REC_CUSTOMER_COMPLAINTS state..")
        print("############################################")
        val = ""
        while val != "0" and val != "1":
            val = input("starting diagnosis with [0] / without [1] customer complaints")

        if val == "0":
            self.launch_customer_xps()
            print("customer XPS session protocol saved..")
            userdata.interview_protocol_file = config.INTERVIEW_PROTOCOL_FILE
            return "complaints_received"
        else:
            print("starting diagnosis without customer complaints..")
            return "no_complaints"


class EstablishInitialHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_hypothesis'],
                             input_keys=['interview_protocol_file'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        print("############################################")
        print("executing ESTABLISH_INITIAL_HYPOTHESIS state..")
        print("############################################")
        print("reading XML protocol..")

        with open(userdata.interview_protocol_file) as f:
            data = f.read()
        session_data = BeautifulSoup(data, 'xml')

        # print(session_data.prettify())

        for tag in session_data.find_all('rating', {'type': 'heuristic'}):
            res = tag.parent['objectName']

        print("establish hypothesis..")
        time.sleep(10)
        userdata.hypothesis = res
        return "established_hypothesis"


class ReadOBDInformation(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_OBD_info'],
                             input_keys=['hypothesis', 'context'],
                             output_keys=['processed_OBD_info'])

    def execute(self, userdata):
        print("############################################")
        print("executing READ_OBD_INFORMATION state..")
        print("############################################")
        if userdata.hypothesis:
            print("already have a hypothesis based on customer constraints:", userdata.hypothesis)
        else:
            print("starting without hypothesis..")
        # TODO: include OBD parser (OBD codes + meta data)
        run_gui()
        # TODO: read from OBD file
        read_dtc = "P0138"
        oqt = OntologyQueryTool()
        measuring_pos = oqt.query_measuring_pos_by_dtc(read_dtc)
        print("determined measuring pos:", measuring_pos)
        userdata.processed_OBD_info = ""
        print("processed OBD information..")
        time.sleep(10)
        return "processed_OBD_info"


class RetrieveHistoricalInfo(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_historical_info'],
                             input_keys=['obd_info'],
                             output_keys=['obd_and_hist_info'])

    def execute(self, userdata):
        print("############################################")
        print("executing RETRIEVE_HISTORICAL_INFO state..")
        print("############################################")
        # TODO: retrieve historical info for the specified vehicle
        #   - using the VIN (obtained from OBD reading)
        #   - use the information to deny certain hypotheses (e.g. repeated replacement of same component)
        #
        # - two kinds of information:
        #   - historical info for specific vehicle (via VIN)
        #   - historical info for vehicle type (model)
        time.sleep(10)
        return "processed_historical_info"


class SuggestMeasuringPos(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_suggestion'],
                             input_keys=['processed_OBD_info'],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing SUGGEST_MEASURING_POS state..")
        print("############################################")
        print(userdata.processed_OBD_info)
        # TODO: implement suggestion for measuring pos
        print("SUGGESTED MEASURING POS: X, Y, Z")
        time.sleep(10)
        return "provided_suggestion"


class PerformSensorRecording(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_sensor_data'],
                             input_keys=[''],
                             output_keys=['oscillogram'])

    def execute(self, userdata):
        print("############################################")
        print("executing PERFORM_SENSOR_RECORDING state..")
        print("############################################")
        # TODO: perform sensor recording and process data -> generate oscillogram
        userdata.oscillogram = ""
        time.sleep(10)
        print("performed sensor recording..")
        return "processed_sensor_data"


class PerformDataManagement(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['data_management_performed'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing PERFORM_DATA_MANAGEMENT state..")
        print("############################################")
        time.sleep(10)
        # TODO:
        #   - EDC (Eclipse Dataspace Connector) communication
        #   - consolidate + upload data (user info, customer complaints, OBD info, sensor data) to server
        #   - optional: [retrieve latest version of trained NN from server]
        return "data_management_performed"


class MapOscillogramToDiagnosis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['diagnosis_determined', 'no_diagnosis', 'no_diagnosis_final'],
                             input_keys=['oscillogram'],
                             output_keys=['diagnosis'])

        self.no_diag_cnt = 0

    def execute(self, userdata):
        print("############################################")
        print("executing MAP_OSCILLOGRAM_TO_DIAGNOSIS state..")
        print("############################################")
        net_input = userdata.oscillogram
        # TODO: apply trained NN
        userdata.diagnosis = ""
        feasible_diag = True
        time.sleep(10)
        print("mapped oscillogram to diagnosis..")
        if feasible_diag:
            return "diagnosis_provided"
        elif self.no_diag_cnt < 3:
            self.no_diag_cnt += 1
            return "no_diagnosis"
        else:
            return "no_diagnosis_final"


class ProvideDiagAndShowTrace(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['info_provided'],
                             input_keys=['diagnosis'],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing PROVIDE_DIAG_AND_SHOW_TRACE state..")
        print("############################################")
        diag = userdata.diagnosis
        # TODO: OPTIONAL: apply [XPS / ...] that recommends action based on diagnosis
        #   - print action
        #   - show trace
        return "info_provided"


class VehicleDiagnosisAndRecommendationStateMachine(smach.StateMachine):

    def __init__(self):
        super(VehicleDiagnosisAndRecommendationStateMachine, self).__init__(
            outcomes=['diag', 'no_diag'],
            input_keys=[],
            output_keys=[]
        )
        self.userdata.sm_input = []

        with self:

            self.add('PROC_USER_INFO', ProcUserInfo(),
                     transitions={'read_user_info': 'REC_CUSTOMER_COMPLAINTS'},
                     remapping={})

            self.add('REC_CUSTOMER_COMPLAINTS', RecCustomerComplaints(),
                     transitions={'complaints_received': 'ESTABLISH_INITIAL_HYPOTHESIS',
                                  'no_complaints': 'READ_OBD_INFORMATION'},
                     remapping={'input_info': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(),
                     transitions={'established_hypothesis': 'READ_OBD_INFORMATION'},
                     remapping={'interview_protocol_file': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('READ_OBD_INFORMATION', ReadOBDInformation(),
                     transitions={'processed_OBD_info': 'RETRIEVE_HISTORICAL_INFO'},
                     remapping={'hypothesis': 'sm_input',
                                'processed_OBD_info': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_INFO', RetrieveHistoricalInfo(),
                     transitions={'processed_historical_info': 'SUGGEST_MEASURING_POS'},
                     remapping={'obd_info': 'sm_input',
                                'obd_and_hist_info': 'sm_input'})

            self.add('SUGGEST_MEASURING_POS', SuggestMeasuringPos(),
                     transitions={'provided_suggestion': 'PERFORM_SENSOR_RECORDING'},
                     remapping={'processed_OBD_info': 'sm_input'})

            self.add('PERFORM_SENSOR_RECORDING', PerformSensorRecording(),
                     transitions={'processed_sensor_data': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'oscillogram': 'sm_input'})

            self.add('PERFORM_DATA_MANAGEMENT', PerformDataManagement(),
                     transitions={'data_management_performed': 'MAP_OSCILLOGRAM_TO_DIAGNOSIS'},
                     remapping={})

            self.add('MAP_OSCILLOGRAM_TO_DIAGNOSIS', MapOscillogramToDiagnosis(),
                     transitions={'diagnosis_determined': 'PROVIDE_DIAG_AND_SHOW_TRACE',
                                  'no_diagnosis_final': 'no_diag',
                                  'no_diagnosis': 'SUGGEST_MEASURING_POS'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(),
                     transitions={'info_provided': 'diag'},
                     remapping={'diagnosis': 'sm_input'})


def node():
    rospy.init_node('test')
    sm = VehicleDiagnosisAndRecommendationStateMachine()
    sis = smach_ros.IntrospectionServer('server_name', sm, '/AW4.0')
    sis.start()
    outcome = sm.execute()
    print("OUTCOME:", outcome)

    rospy.spin()
    sis.stop()


if __name__ == '__main__':
    try:
        node()
    except rospy.ROSInterruptException:
        pass

