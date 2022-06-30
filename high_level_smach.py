#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach
import subprocess
from bs4 import BeautifulSoup


class RecCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['complaints_received', 'no_complaints'],
                             input_keys=['input_info'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        print("launching customer XPS..")
        subprocess.call(['java', '-jar', 'CustomerXPS.jar'])

    def execute(self, userdata):
        print("executing REC_CUSTOMER_COMPLAINTS state..")
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


class EstablishContextOrHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_context', 'established_hypothesis', 'unknown'],
                             input_keys=['interview_protocol_file'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        print("executing ESTABLISH_CONTEXT_OR_HYPOTHESIS state..")
        print("reading XML protocol..")

        with open(userdata.interview_protocol_file) as f:
            data = f.read()
        session_data = BeautifulSoup(data, 'xml')

        # print(session_data.prettify())

        for tag in session_data.find_all('rating', {'type': 'heuristic'}):
            res = tag.parent['objectName']

        userdata.hypothesis = res
        return "established_hypothesis"


class ReadOBDInformation(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_OBD_info'],
                             input_keys=['hypothesis', 'context'],
                             output_keys=['processed_OBD_info'])

    def execute(self, userdata):
        print("executing READ_OBD_INFORMATION state..")
        if userdata.hypothesis:
            print("already have a hypothesis based on customer constraints:", userdata.hypothesis)
        else:
            print("starting without hypothesis..")
        # TODO: include OBD parser (OBD codes + meta data)
        userdata.processed_OBD_info = ""
        return "processed_OBD_info"


class SuggestMeasuringPos(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_suggestion'],
                             input_keys=['processed_OBD_info'],
                             output_keys=[''])

    def execute(self, userdata):
        print("executing SUGGEST_MEASURING_POS state..")
        print(userdata.processed_OBD_info)
        # TODO: implement suggestion for measuring pos
        return "provided_suggestion"


class PerformSensorRecording(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_sensor_data'],
                             input_keys=[''],
                             output_keys=['oscillogram'])

    def execute(self, userdata):
        print("executing PERFORM_SENSOR_RECORDING state..")
        # TODO: perform sensor recording and process data -> generate oscillogram
        userdata.oscillogram = ""
        return "processed_sensor_data"


class MapOscillogramToDiagnosis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['diagnosis_provided'],
                             input_keys=['oscillogram'],
                             output_keys=['diagnosis'])

    def execute(self, userdata):
        print("executing MAP_OSCILLOGRAM_TO_DIAGNOSIS state..")
        net_input = userdata.oscillogram
        # TODO: apply trained NN
        userdata.diagnosis = ""
        return "diagnosis_provided"


class RecommendActionAndShowTrace(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['action_suggestion_and_trace_provided'],
                             input_keys=['diagnosis'],
                             output_keys=[''])

    def execute(self, userdata):
        print("executing RECOMMEND_ACTION_AND_SHOW_TRACE state..")
        diag = userdata.diagnosis
        # TODO: apply [XPS / ...] that recommends action based on diagnosis
        #   - print action
        #   - show trace
        return "action_suggestion_and_trace_provided"


class VehicleDiagnosisAndRecommendationStateMachine(smach.StateMachine):

    def __init__(self):
        super(VehicleDiagnosisAndRecommendationStateMachine, self).__init__(
            outcomes=['diagnosis_without_recommendation', 'diagnosis_with_recommendation', 'no_diagnosis'],
            input_keys=[],
            output_keys=[]
        )
        self.userdata.sm_input = []

        with self:
            self.add('REC_CUSTOMER_COMPLAINTS', RecCustomerComplaints(),
                     transitions={'complaints_received': 'ESTABLISH_CONTEXT_OR_HYPOTHESIS',
                                  'no_complaints': 'READ_OBD_INFORMATION'},
                     remapping={'input_info': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('ESTABLISH_CONTEXT_OR_HYPOTHESIS', EstablishContextOrHypothesis(),
                     transitions={'established_context': 'READ_OBD_INFORMATION',
                                  'established_hypothesis': 'READ_OBD_INFORMATION',
                                  'unknown': 'READ_OBD_INFORMATION'},
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
                     transitions={'diagnosis_provided': 'RECOMMEND_ACTION_AND_SHOW_TRACE'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('RECOMMEND_ACTION_AND_SHOW_TRACE', RecommendActionAndShowTrace(),
                     transitions={'action_suggestion_and_trace_provided': 'diagnosis_with_recommendation'},
                     remapping={'diagnosis': 'sm_input'})


if __name__ == '__main__':
    sm = VehicleDiagnosisAndRecommendationStateMachine()
    outcome = sm.execute()
    print("OUTCOME:", outcome)
