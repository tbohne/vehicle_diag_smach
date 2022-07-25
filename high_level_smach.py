#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import subprocess
import sys
import time
from os import path

import rospy
import smach
import smach_ros
from bs4 import BeautifulSoup

import config
from ontology_instance_generator import OntologyInstanceGenerator

sys.path.append(path.abspath('../AW_40_GUI'))
sys.path.append(path.abspath('../OBDOntology'))

from GUI import run_gui

ONTOLOGY_PATH = "../OBDOntology"
ONTOLOGY_FILE = "obd_ontology.owl"


class RecVehicleAndProcUserData(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_user_data'],
                             input_keys=[''],
                             output_keys=['user_data'])

    def execute(self, userdata):
        print("############################################")
        print("executing REC_VEHICLE_AND_PROC_USER_DATA state..")
        print("############################################")
        time.sleep(10)
        userdata.user_data = "dummy user info"
        return "processed_user_data"


class ProcCustomerComplaints(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['received_complaints', 'no_complaints'],
                             input_keys=['user_data'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        print("launching customer XPS..")
        subprocess.call(['java', '-jar', config.CUSTOMER_XPS])

    def execute(self, userdata):
        print("############################################")
        print("executing PROC_CUSTOMER_COMPLAINTS state..")
        print("############################################")

        print("provided user data:", userdata.user_data)

        val = ""
        while val != "0" and val != "1":
            val = input("starting diagnosis with [0] / without [1] customer complaints")

        if val == "0":
            self.launch_customer_xps()
            print("customer XPS session protocol saved..")
            userdata.interview_protocol_file = config.INTERVIEW_PROTOCOL_FILE
            return "received_complaints"
        else:
            print("starting diagnosis without customer complaints..")
            userdata.interview_protocol_file = ""
            return "no_complaints"


class EstablishInitialHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_init_hypothesis', 'no_oscilloscope_required'],
                             input_keys=['interview_protocol_file'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        print("############################################")
        print("executing ESTABLISH_INITIAL_HYPOTHESIS state..")
        print("############################################")
        print("reading XML protocol..")

        if userdata.interview_protocol_file:
            print("customer complaints available..")
            with open(userdata.interview_protocol_file) as f:
                data = f.read()
            session_data = BeautifulSoup(data, 'xml')
            # print(session_data.prettify())
            for tag in session_data.find_all('rating', {'type': 'heuristic'}):
                res = tag.parent['objectName']

            userdata.hypothesis = res
        else:
            print("no customer complaints available..")

        oscilloscope_required = True
        if not oscilloscope_required:
            return "no_oscilloscope_required"

        print("establish hypothesis..")
        time.sleep(10)
        return "established_init_hypothesis"


class ReadOBDDataAndGenOntologyInstance(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_OBD_data', 'no_OBD_data'],
                             input_keys=['interview_data'],
                             output_keys=['processed_OBD_data'])

    def execute(self, userdata):
        print("############################################")
        print("executing READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCE state..")
        print("############################################")

        print("OBD INPUT:", userdata.interview_data)

        # TODO: include OBD parser (OBD codes + meta data)
        run_gui()
        obd_avail = True
        if not obd_avail:
            return "no_OBD_data"

        # OBD data available
        # TODO: read from OBD file
        read_dtc = "P1111"
        read_model = "Mazda 3"
        read_hsn = "849357984"
        read_tsn = "453948539"
        read_vin = "1234567890ABCDEFGHJKLMNPRSTUVWXYZ"
        time.sleep(10)
        print("processed OBD information..")

        # TODO: first step is a lookup on our own server
        #   - did we create an ontology instance for this vehicle-DTC combination before?
        #   - if so, retrieve this instance instead of creating a new one from external DB data
        instance_match_on_server = False
        if instance_match_on_server:
            pass
        else:
            # generate ontology instance based on read OBD data
            instance_gen = OntologyInstanceGenerator(
                read_model, read_hsn, read_tsn, read_vin, read_dtc, ONTOLOGY_PATH, ONTOLOGY_FILE
            )
            instance_gen.create_ontology_instance()

        userdata.processed_OBD_data = userdata.interview_data
        return "processed_OBD_data"


class RetrieveHistoricalData(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_all_data'],
                             input_keys=['obd_info'],
                             output_keys=['obd_and_hist_info'])

    def execute(self, userdata):
        print("############################################")
        print("executing RETRIEVE_HISTORICAL_DATA state..")
        print("############################################")
        # TODO: retrieve historical info for the specified vehicle
        #   - using the VIN (obtained from OBD reading)
        #   - use the information to deny certain hypotheses (e.g. repeated replacement of same component)
        #
        # - two kinds of information:
        #   - historical info for specific vehicle (via VIN)
        #   - historical info for vehicle type (model)
        userdata.obd_and_hist_info = userdata.obd_info
        time.sleep(10)
        return "processed_all_data"


class SuggestMeasuringPos(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_suggestion'],
                             input_keys=['processed_OBD_data'],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing SUGGEST_MEASURING_POS state..")
        print("############################################")
        print(userdata.processed_OBD_data)
        # TODO: implement suggestion for measuring pos
        print("SUGGESTED MEASURING POS: X, Y, Z")
        # oqt = OntologyQueryTool()
        # measuring_pos = oqt.query_measuring_pos_by_dtc(read_dtc)
        # print("determined measuring pos:", measuring_pos)
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
                             outcomes=['performed_data_management', 'performed_reduced_data_management'],
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

        no_oscilloscope = True
        if no_oscilloscope:
            return "performed_reduced_data_management"
        return "performed_data_management"


class MapOscillogramToSymptom(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['determined_symptom', 'no_mapping', 'conclusively_no_mapping'],
                             input_keys=['oscillogram'],
                             output_keys=['diagnosis'])

        self.no_diag_cnt = 0

    def execute(self, userdata):
        print("############################################")
        print("executing MAP_OSCILLOGRAM_TO_SYMPTOM state..")
        print("############################################")
        net_input = userdata.oscillogram
        # TODO: apply trained NN
        userdata.diagnosis = ""
        feasible_diag = True
        time.sleep(10)
        print("mapped oscillogram to diagnosis..")
        if feasible_diag:
            return "determined_symptom"
        elif self.no_diag_cnt < 3:
            self.no_diag_cnt += 1
            return "no_mapping"
        else:
            return "conclusively_no_mapping"


class ProvideDiagAndShowTrace(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_diag_and_explanation'],
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
        return "provided_diag_and_explanation"


class ProvideInitialHypothesis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_limited_diag'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing PROVIDE_INITIAL_HYPOTHESIS state..")
        print("############################################")
        return "provided_limited_diag"


class UploadDiagnosis(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['uploaded_limited_diag', 'uploaded_diag'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        print("############################################")
        print("executing UPLOAD_DIAGNOSIS state..")
        print("############################################")
        return "uploaded_diag"


class VehicleDiagnosisAndRecommendationStateMachine(smach.StateMachine):

    def __init__(self):
        super(VehicleDiagnosisAndRecommendationStateMachine, self).__init__(
            outcomes=['diag', 'lim_diag'],
            input_keys=[],
            output_keys=[]
        )
        self.userdata.sm_input = []

        with self:

            self.add('REC_VEHICLE_AND_PROC_USER_DATA', RecVehicleAndProcUserData(),
                     transitions={'processed_user_data': 'PROC_CUSTOMER_COMPLAINTS'},
                     remapping={'input': 'sm_input',
                                'user_data': 'sm_input'})

            self.add('PROC_CUSTOMER_COMPLAINTS', ProcCustomerComplaints(),
                     transitions={'received_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCE',
                                  'no_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCE'},
                     remapping={'user_data': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCE', ReadOBDDataAndGenOntologyInstance(),
                     transitions={'processed_OBD_data': 'RETRIEVE_HISTORICAL_DATA',
                                  'no_OBD_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'interview_data': 'sm_input',
                                'processed_OBD_data': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(),
                     transitions={'established_init_hypothesis': 'SUGGEST_MEASURING_POS',
                                  'no_oscilloscope_required': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'interview_protocol_file': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_DATA', RetrieveHistoricalData(),
                     transitions={'processed_all_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'obd_info': 'sm_input',
                                'obd_and_hist_info': 'sm_input'})

            self.add('SUGGEST_MEASURING_POS', SuggestMeasuringPos(),
                     transitions={'provided_suggestion': 'PERFORM_SENSOR_RECORDING'},
                     remapping={'processed_OBD_data': 'sm_input'})

            self.add('PERFORM_SENSOR_RECORDING', PerformSensorRecording(),
                     transitions={'processed_sensor_data': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'oscillogram': 'sm_input'})

            self.add('PERFORM_DATA_MANAGEMENT', PerformDataManagement(),
                     transitions={'performed_data_management': 'MAP_OSCILLOGRAM_TO_SYMPTOM',
                                  'performed_reduced_data_management': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('MAP_OSCILLOGRAM_TO_SYMPTOM', MapOscillogramToSymptom(),
                     transitions={'determined_symptom': 'PROVIDE_DIAG_AND_SHOW_TRACE',
                                  'conclusively_no_mapping': 'PROVIDE_INITIAL_HYPOTHESIS',
                                  'no_mapping': 'SUGGEST_MEASURING_POS'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(),
                     transitions={'provided_diag_and_explanation': 'UPLOAD_DIAGNOSIS'},
                     remapping={'diagnosis': 'sm_input'})

            self.add('PROVIDE_INITIAL_HYPOTHESIS', ProvideInitialHypothesis(),
                     transitions={'provided_limited_diag': 'UPLOAD_DIAGNOSIS'},
                     remapping={})

            self.add('UPLOAD_DIAGNOSIS', UploadDiagnosis(),
                     transitions={'uploaded_limited_diag': 'lim_diag',
                                  'uploaded_diag': 'diag'},
                     remapping={})


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
