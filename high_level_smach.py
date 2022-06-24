#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach
import time


class RecCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['waiting_for_complaints', 'complaints_received'],
                             input_keys=['input_info'],
                             output_keys=['proc_customer_complaints'])

    def get_complaints(self):
        return ""

    def execute(self, userdata):
        print("executing REC_CUSTOMER_COMPLAINTS state..")

        if len(userdata.input_info) > 0:
            print("input_info: %s", userdata.input_info)
            userdata.proc_customer_complaints = userdata.input_info
            return "complaints_received"

        print("waiting for customer complaints..")
        complaints = self.get_complaints()

        if complaints:
            print("received complaints..")
            if len(complaints) == 0:
                print("empty complaints..")
                return "waiting_for_complaints"

            userdata.proc_customer_complaints = complaints
            return "complaints_received"
        else:
            print("waiting for complaints..")
            time.sleep(10)
            return "waiting_for_complaints"


class EstablishContextOrHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_context', 'established_hypothesis', 'unknown'],
                             input_keys=['proc_customer_complaints'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        print("executing ESTABLISH_CONTEXT_OR_HYPOTHESIS state..")

        userdata.hypothesis = "failure X"
        return 'hypothesis'


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
                                  'complaints_received': 'ESTABLISH_CONTEXT_OR_HYPOTHESIS'},
                     remapping={'input_info': 'sm_input',
                                'proc_customer_complaints': 'sm_input'})

            self.add('ESTABLISH_CONTEXT_OR_HYPOTHESIS', EstablishContextOrHypothesis(),
                     transitions={'established_context': 'no_diagnosis',
                                  'established_hypothesis': 'diagnosis_without_recommendation',
                                  'unknown': 'no_diagnosis'},
                     remapping={'hypothesis': 'sm_input'})


if __name__ == '__main__':
    sm = VehicleDiagnosisAndRecommendationStateMachine()
    outcome = sm.execute()
    print("OUTCOME:", outcome)
