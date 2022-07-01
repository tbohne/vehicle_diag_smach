#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach
import subprocess
import time
from bs4 import BeautifulSoup

from os import path
import sys
sys.path.append(path.abspath('../AW_40_GUI'))

from GUI import run_gui


class RecCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['complaints_received', 'no_complaints'],
                             input_keys=['input_info'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        print("launching customer XPS..")
        subprocess.call(['java', '-jar', 'lib/CustomerXPS.jar'])

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
            userdata.interview_protocol_file = "./KB/session_res.xml"
            return "complaints_received"
        else:
            print("starting diagnosis without customer complaints..")
            return "no_complaints"


class EstablishHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_hypothesis'],
                             input_keys=['interview_protocol_file'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        print("############################################")
        print("executing ESTABLISH_HYPOTHESIS state..")
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

        userdata.processed_OBD_info = ""
        print("processed OBD information..")
        time.sleep(10)
        return "processed_OBD_info"


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


class MapOscillogramToDiagnosis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['diagnosis_provided', 'no_diagnosis'],
                             input_keys=['oscillogram'],
                             output_keys=['diagnosis'])

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
        return "no_diagnosis"


class RecommendActionAndShowTrace(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['info_provided', 'no_suggestion'],
                             input_keys=['diagnosis'],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing RECOMMEND_ACTION_AND_SHOW_TRACE state..")
        print("############################################")
        diag = userdata.diagnosis
        # TODO: apply [XPS / ...] that recommends action based on diagnosis
        #   - print action
        #   - show trace
        feasible_suggestion = True
        time.sleep(10)
        print("recommended action and showed trace..")
        if feasible_suggestion:
            return "info_provided"
        return "no_suggestion"


class VehicleDiagnosisAndRecommendationStateMachine(smach.StateMachine):

    def __init__(self):
        super(VehicleDiagnosisAndRecommendationStateMachine, self).__init__(
            outcomes=['diag_without_reco', 'diag_with_reco', 'no_diag'],
            input_keys=[],
            output_keys=[]
        )
        self.userdata.sm_input = []

        with self:
            self.add('REC_CUSTOMER_COMPLAINTS', RecCustomerComplaints(),
                     transitions={'complaints_received': 'ESTABLISH_HYPOTHESIS',
                                  'no_complaints': 'READ_OBD_INFORMATION'},
                     remapping={'input_info': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('ESTABLISH_HYPOTHESIS', EstablishHypothesis(),
                     transitions={'established_hypothesis': 'READ_OBD_INFORMATION'},
                     remapping={'interview_protocol_file': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('READ_OBD_INFORMATION', ReadOBDInformation(),
                     transitions={'processed_OBD_info': 'SUGGEST_MEASURING_POS'},
                     remapping={'hypothesis': 'sm_input',
                                'processed_OBD_info': 'sm_input'})

            self.add('SUGGEST_MEASURING_POS', SuggestMeasuringPos(),
                     transitions={'provided_suggestion': 'PERFORM_SENSOR_RECORDING'},
                     remapping={'processed_OBD_info': 'sm_input'})

            self.add('PERFORM_SENSOR_RECORDING', PerformSensorRecording(),
                     transitions={'processed_sensor_data': 'MAP_OSCILLOGRAM_TO_DIAGNOSIS'},
                     remapping={'oscillogram': 'sm_input'})

            self.add('MAP_OSCILLOGRAM_TO_DIAGNOSIS', MapOscillogramToDiagnosis(),
                     transitions={'diagnosis_provided': 'RECOMMEND_ACTION_AND_SHOW_TRACE',
                                  'no_diagnosis': 'no_diag'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('RECOMMEND_ACTION_AND_SHOW_TRACE', RecommendActionAndShowTrace(),
                     transitions={'info_provided': 'diag_with_reco',
                                  'no_suggestion': 'diag_without_reco'},
                     remapping={'diagnosis': 'sm_input'})


if __name__ == '__main__':
    sm = VehicleDiagnosisAndRecommendationStateMachine()
    outcome = sm.execute()
    print("OUTCOME:", outcome)

