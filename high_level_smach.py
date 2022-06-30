#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach
import subprocess
from bs4 import BeautifulSoup


class RecCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['waiting_for_complaints', 'complaints_received', 'no_complaints'],
                             input_keys=['input_info'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        print("launching customer XPS..")
        subprocess.call(['java', '-jar', 'CustomerXPS.jar'])

    def execute(self, userdata):
        print("executing REC_CUSTOMER_COMPLAINTS state..")
        self.launch_customer_xps()
        print("customer XPS session protocol saved..")
        userdata.interview_protocol_file = "./KB/session_res.xml"
        return "complaints_received"


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
        return 'established_hypothesis'


class ReadOBDInformation(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['read_OBD_info'],
                             input_keys=['hypothesis', 'context'],
                             output_keys=['measuring_points'])

    def execute(self, userdata):
        print("executing READ_OBD_INFORMATION state..")
        return 'read_OBD_info'


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
                     transitions={'waiting_for_complaints': 'REC_CUSTOMER_COMPLAINTS',
                                  'complaints_received': 'ESTABLISH_CONTEXT_OR_HYPOTHESIS',
                                  'no_complaints': 'READ_OBD_INFORMATION'},
                     remapping={'input_info': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('ESTABLISH_CONTEXT_OR_HYPOTHESIS', EstablishContextOrHypothesis(),
                     transitions={'established_context': 'no_diagnosis',
                                  'established_hypothesis': 'diagnosis_without_recommendation',
                                  'unknown': 'no_diagnosis'},
                     remapping={'interview_protocol_file': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('READ_OBD_INFORMATION', ReadOBDInformation(),
                     transitions={'read_OBD_info': 'no_diagnosis'},
                     remapping={})


if __name__ == '__main__':
    sm = VehicleDiagnosisAndRecommendationStateMachine()
    outcome = sm.execute()
    print("OUTCOME:", outcome)
