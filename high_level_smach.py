#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from datetime import date

import numpy as np
import smach
from AW_40_GUI import GUI
from OBDOntology import ontology_instance_generator
from bs4 import BeautifulSoup
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from py4j.java_gateway import JavaGateway
from tensorflow import keras
from vehicle_diag_smach import config


class RecVehicleAndProcUserData(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic receives the vehicle and processes
    the user data (data about the workshop, mechanic, etc.).
    """

    def __init__(self):
        smach.State.__init__(self, outcomes=['processed_user_data'], input_keys=[''], output_keys=['user_data'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'REC_VEHICLE_AND_PROC_USER_DATA' state.

        :param userdata:  input of state
        :return: outcome of the state ("processed_user_data")
        """
        print("############################################")
        print("executing REC_VEHICLE_AND_PROC_USER_DATA state..")
        print("############################################")

        # TODO: read from updated GUI
        user_data = {
            "workshop_name": "workshop_one",
            "zipcode": "12345",
            "workshop_id": "00000",
            "mechanic_id": "99999",
            # how many parallel measurements are possible at most (based on workshop equipment / oscilloscope channels)
            "max_number_of_parallel_recordings": "1",
            "date": date.today()
        }

        # if not present, create directory for session data
        if not os.path.exists(config.SESSION_DIR):
            print("creating session data directory..")
            os.makedirs(config.SESSION_DIR)

        # write user data to session directory
        with open(config.SESSION_DIR + '/user_data.json', 'w') as f:
            json.dump(user_data, f, default=str)

        return "processed_user_data"


class ProcCustomerComplaints(smach.State):
    """
    State in the high-level SMACH that represents situations in which the mechanic enters the customer complaints
    to the processing system (fault tree, decision tree, XPS, ...).
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['received_complaints', 'no_complaints'],
                             input_keys=['user_data'],
                             output_keys=['interview_protocol_file'])

    @staticmethod
    def launch_customer_xps():
        """
        Launches the expert system that processes the customer complaints.
        """
        print("establish connection to customer XPS server..")
        gateway = JavaGateway()
        customer_xps = gateway.entry_point
        print("result of customer xps: ", customer_xps.demo())

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROC_CUSTOMER_COMPLAINTS' state.

        :param userdata: input of state (provided user data)
        :return: outcome of the state ("received_complaints" | "no_complaints")
        """
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
    """
    State in the high-level SMACH that represents situations in which an initial hypothesis is established based
    on the provided information.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['established_init_hypothesis', 'no_oscilloscope_required', 'no_OBD_and_no_CC'],
                             input_keys=['interview_protocol_file'],
                             output_keys=['context', 'hypothesis'])

    def execute(self, userdata):
        """
        Execution of 'ESTABLISH_INITIAL_HYPOTHESIS' state.

        :param userdata:  input of state
        :return: outcome of the state ("established_init_hypothesis" | "no_oscilloscope_required" | "no_OBD_and_no_CC")
        """
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
            print("RES:", res)

            userdata.hypothesis = res
        else:
            print("no customer complaints available..")

        oscilloscope_required = True
        if not oscilloscope_required:
            return "no_oscilloscope_required"

        print("establish hypothesis..")
        # time.sleep(10)
        return "established_init_hypothesis"


class ReadOBDDataAndGenOntologyInstances(smach.State):
    """
    State in the high-level SMACH that represents situations in which the OBD information are read from the ECU.
    Based on the read information, an ontology instance is generated.
    TODO: actually, based on the current state, we will probably have ontology instances generated in advance by experts
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_OBD_data', 'no_OBD_data'],
                             input_keys=['interview_data'],
                             output_keys=['processed_OBD_data'])

    def execute(self, userdata):
        """
        Execution of 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata:  input of state
        :return: outcome of the state ("processed_OBD_data" | "no_OBD_data")
        """
        print("############################################")
        print("executing READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES state..")
        print("############################################")

        print("OBD INPUT:", userdata.interview_data)

        # TODO: include OBD parser (OBD codes + meta data)
        GUI.run_gui()
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
        # time.sleep(10)
        print("processed OBD information..")

        # TODO: first step is a lookup on our own server
        #   - did we create an ontology instance for this vehicle-DTC combination before?
        #   - if so, retrieve this instance instead of creating a new one from external DB data
        instance_match_on_server = False
        if instance_match_on_server:
            pass
        else:
            # generate ontology instance based on read OBD data
            instance_gen = ontology_instance_generator.OntologyInstanceGenerator(
                config.OBD_ONTOLOGY_PATH, local_kb=False
            )
            instance_gen.extend_knowledge_graph(read_model, read_hsn, read_tsn, read_vin, read_dtc)

        userdata.processed_OBD_data = userdata.interview_data
        return "processed_OBD_data"


class RetrieveHistoricalData(smach.State):
    """
    State in the high-level SMACH that represents situations in which historical information are retrieved for the
    given car (individually, not type), i.e., information that we accumulated in previous repair sessions in the past.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_all_data'],
                             input_keys=['obd_info'],
                             output_keys=['obd_and_hist_info'])

    def execute(self, userdata):
        """
        Execution of 'RETRIEVE_HISTORICAL_DATA' state.

        :param userdata:  input of state
        :return: outcome of the state ("processed_all_data")
        """
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
        # time.sleep(10)
        return "processed_all_data"


class SuggestMeasuringPosOrComponents(smach.State):
    """
    State in the high-level SMACH that represents situations in which measuring positions or at least suspect
    components in the car are suggested in order to place the sensors (oscilloscopes) accordingly.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_suggestions'],
                             input_keys=['processed_OBD_data'],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'SUGGEST_MEASURING_POS_OR_COMPONENTS' state.

        :param userdata:  input of state
        :return: outcome of the state ("provided_suggestions")
        """
        print("############################################")
        print("executing SUGGEST_MEASURING_POS_OR_COMPONENTS state..")
        print("############################################")
        print(userdata.processed_OBD_data)
        # TODO: implement suggestion for measuring pos
        print("SUGGESTED MEASURING POS: X, Y, Z")
        # oqt = OntologyQueryTool()
        # measuring_pos = oqt.query_measuring_pos_by_dtc(read_dtc)
        # print("determined measuring pos:", measuring_pos)
        # time.sleep(10)
        return "provided_suggestions"


class PerformSynchronizedSensorRecordings(smach.State):
    """
    State in the high-level SMACH that represents situations in which the synchronized sensor recordings are performed.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_sync_sensor_data'],
                             input_keys=[''],
                             output_keys=['oscillogram'])

    def execute(self, userdata):
        """
        Execution of 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS' state.

        :param userdata:  input of state
        :return: outcome of the state ("processed_sync_sensor_data")
        """
        print("############################################")
        print("executing PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS state..")
        print("############################################")
        # TODO: perform sensor recording and process data -> generate oscillogram
        userdata.oscillogram = ""
        # time.sleep(10)
        print("performed sensor recordings..")
        return "processed_sync_sensor_data"


class PerformDataManagement(smach.State):
    """
    State in the high-level SMACH that represents situations in which data management is performed.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['performed_data_management', 'performed_reduced_data_management'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'PERFORM_DATA_MANAGEMENT' state.

        :param userdata:  input of state
        :return: outcome of the state ("performed_data_management" | "performed_reduced_data_management")
        """
        print("############################################")
        print("executing PERFORM_DATA_MANAGEMENT state..")
        print("############################################")
        # time.sleep(10)
        # TODO:
        #   - EDC (Eclipse Dataspace Connector) communication
        #   - consolidate + upload data (user info, customer complaints, OBD info, sensor data) to server
        #   - optional: [retrieve latest version of trained NN from server]

        no_oscilloscope = False
        if no_oscilloscope:
            return "performed_reduced_data_management"
        return "performed_data_management"


class ClassifyOscillograms(smach.State):
    """
    State in the high-level SMACH that represents situations in which the recorded oscillograms are classified using
    the trained neural net model, i.e., detecting anomalies.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['detected_anomalies', 'no_anomaly',
                                       'no_anomaly_and_no_more_measuring_pos'],
                             input_keys=['oscillogram'],
                             output_keys=['diagnosis'])

    def execute(self, userdata):
        """
        Execution of 'CLASSIFY_OSCILLOGRAMS' state.

        :param userdata:  input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_and_no_more_measuring_pos")
        """
        print("############################################")
        print("executing CLASSIFY_OSCILLOGRAMS state (apply trained CNN)..")
        print("############################################")

        # net_input = userdata.oscillogram

        _, voltages = preprocess.read_oscilloscope_recording(config.DUMMY_OSCILLOSCOPE)
        voltages = preprocess.z_normalize_time_series(voltages)

        model = keras.models.load_model(config.TRAINED_MODEL)

        # fix input size
        net_input_size = model.layers[0].output_shape[0][1]
        if len(voltages) > net_input_size:
            remove = len(voltages) - net_input_size
            voltages = voltages[: len(voltages) - remove]

        net_input = np.asarray(voltages).astype('float32')
        net_input = net_input.reshape((net_input.shape[0], 1))

        print("input shape:", net_input.shape)

        prediction = model.predict(np.array([net_input]))
        print("PREDICTION:", prediction)
        print("shape of pred.:", prediction.shape)

        heatmaps = {'gradcam': cam.generate_gradcam(np.array([net_input]), model)}
        cam.plot_heatmaps(heatmaps, voltages)

        userdata.diagnosis = ""
        at_least_one_anomaly = True
        remaining_measuring_pos_suggestions = True
        # time.sleep(10)
        print("mapped oscillogram to diagnosis..")
        if at_least_one_anomaly:
            return "detected_anomalies"
        elif remaining_measuring_pos_suggestions:
            return "no_anomaly"
        else:
            return "no_anomaly_and_no_more_measuring_pos"


class ProvideDiagAndShowTrace(smach.State):
    """
    State in the high-level SMACH that represents situations in which the diagnosis is provided in combination with
    a detailed trace of all the relevant information that lead to it.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['provided_diag_and_explanation'],
                             input_keys=['diagnosis'],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        :param userdata:  input of state
        :return: outcome of the state ("provided_diag_and_explanation")
        """
        print("############################################")
        print("executing PROVIDE_DIAG_AND_SHOW_TRACE state..")
        print("############################################")
        diag = userdata.diagnosis
        # TODO: OPTIONAL: apply [XPS / ...] that recommends action based on diagnosis
        #   - print action
        #   - show trace
        return "provided_diag_and_explanation"


class ProvideInitialHypothesis(smach.State):
    """
    State in the high-level SMACH that represents situations in which only the initial hypothesis is provided due to
    unmanageable uncertainty.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['no_diag'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS' state.

        :param userdata:  input of state
        :return: outcome of the state ("no_diag")
        """
        print("############################################")
        print("executing PROVIDE_INITIAL_HYPOTHESIS state..")
        print("############################################")
        return "no_diag"


class UploadDiagnosis(smach.State):
    """
    State in the high-level SMACH that represents situations in which the diagnosis is uploaded to the server.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['uploaded_diag'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'UPLOAD_DIAGNOSIS' state.

        :param userdata:  input of state
        :return: outcome of the state ("uploaded_diag")
        """
        print("############################################")
        print("executing UPLOAD_DIAGNOSIS state..")
        print("############################################")
        return "uploaded_diag"


class GenArtificialInstanceBasedOnCC(smach.State):
    """
    State in the high-level SMACH that represents situations in which an artificial OBD ontology instance is generated
    based on the customer complaints. Used for cases where no OBD information is available.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['generated_artificial_instance'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC' state.

        :param userdata:  input of state
        :return: outcome of the state ("generated_artificial_instance")
        """
        print("############################################")
        print("executing GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC state..")
        print("############################################")
        return "generated_artificial_instance"


class SelectBestUnusedInstance(smach.State):
    """
    State in the high-level SMACH that represents situations in which a best-suited, unused ontology instance is
    selected for further processing.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['selected_matching_instance(OBD_CC)', 'no_matching_selected_best_instance',
                                       'no_instance', 'no_instance_and_CC_already_used'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'SELECT_BEST_UNUSED_INSTANCE' state.

        :param userdata:  input of state
        :return: outcome of the state ("selected_matching_instance(OBD_CC)" | "no_matching_selected_best_instance" |
                                       "no_instance" | "no_instance_and_CC_already_used")
        """
        print("############################################")
        print("executing SELECT_BEST_UNUSED_INSTANCE state..")
        print("############################################")
        return "selected_matching_instance(OBD_CC)"


class NoProblemDetectedCheckSensor(smach.State):
    """
    State in the high-level SMACH that represents situations in which no actual anomaly was detected, and the indirect
    conclusion of a potential sensor malfunction is provided. This conclusion should be verified / refuted in this
    state.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['sensor_works', 'sensor_defective'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'NO_PROBLEM_DETECTED_CHECK_SENSOR' state.

        :param userdata:  input of state
        :return: outcome of the state ("sensor_works" | "sensor_defective")
        """
        print("############################################")
        print("executing NO_PROBLEM_DETECTED_CHECK_SENSOR state..")
        print("############################################")
        return "sensor_works"


class IsolateProblemCheckEffectiveRadius(smach.State):
    """
    State in the high-level SMACH that represents situations in which one or more anomalies have been detected, and the
    task is to isolate the defective components based on the effective radius (structural knowledge).
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['isolated_problem'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS' state.

        :param userdata:  input of state
        :return: outcome of the state ("isolated_problem")
        """
        print("############################################")
        print("executing ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS state..")
        print("############################################")
        return "isolated_problem"


class VehicleDiagnosisStateMachine(smach.StateMachine):
    """
    High-level state machine guiding the entire vehicle diagnosis process.
    """

    def __init__(self):
        super(VehicleDiagnosisStateMachine, self).__init__(
            outcomes=['diag', 'insufficient_data', 'refuted_hypothesis'],
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
                     transitions={'received_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES',
                                  'no_complaints': 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES'},
                     remapping={'user_data': 'sm_input',
                                'interview_protocol_file': 'sm_input'})

            self.add('READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES', ReadOBDDataAndGenOntologyInstances(),
                     transitions={'processed_OBD_data': 'RETRIEVE_HISTORICAL_DATA',
                                  'no_OBD_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'interview_data': 'sm_input',
                                'processed_OBD_data': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(),
                     transitions={'established_init_hypothesis': 'SELECT_BEST_UNUSED_INSTANCE',
                                  'no_oscilloscope_required': 'PERFORM_DATA_MANAGEMENT',
                                  'no_OBD_and_no_CC': 'insufficient_data'},
                     remapping={'interview_protocol_file': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_DATA', RetrieveHistoricalData(),
                     transitions={'processed_all_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'obd_info': 'sm_input',
                                'obd_and_hist_info': 'sm_input'})

            self.add('SUGGEST_MEASURING_POS_OR_COMPONENTS', SuggestMeasuringPosOrComponents(),
                     transitions={'provided_suggestions': 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS'},
                     remapping={'processed_OBD_data': 'sm_input'})

            self.add('PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS', PerformSynchronizedSensorRecordings(),
                     transitions={'processed_sync_sensor_data': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'oscillogram': 'sm_input'})

            self.add('PERFORM_DATA_MANAGEMENT', PerformDataManagement(),
                     transitions={'performed_data_management': 'CLASSIFY_OSCILLOGRAMS',
                                  'performed_reduced_data_management': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('CLASSIFY_OSCILLOGRAMS', ClassifyOscillograms(),
                     transitions={'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_INSTANCE',
                                  'no_anomaly': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(),
                     transitions={'provided_diag_and_explanation': 'UPLOAD_DIAGNOSIS'},
                     remapping={'diagnosis': 'sm_input'})

            self.add('PROVIDE_INITIAL_HYPOTHESIS', ProvideInitialHypothesis(),
                     transitions={'no_diag': 'refuted_hypothesis'},
                     remapping={})

            self.add('UPLOAD_DIAGNOSIS', UploadDiagnosis(),
                     transitions={'uploaded_diag': 'diag'},
                     remapping={})

            self.add('GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC', GenArtificialInstanceBasedOnCC(),
                     transitions={'generated_artificial_instance': 'SUGGEST_MEASURING_POS_OR_COMPONENTS'},
                     remapping={})

            self.add('SELECT_BEST_UNUSED_INSTANCE', SelectBestUnusedInstance(),
                     transitions={'selected_matching_instance(OBD_CC)': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'no_matching_selected_best_instance': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'no_instance': 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC',
                                  'no_instance_and_CC_already_used': 'NO_PROBLEM_DETECTED_CHECK_SENSOR'},
                     remapping={})

            self.add('NO_PROBLEM_DETECTED_CHECK_SENSOR', NoProblemDetectedCheckSensor(),
                     transitions={'sensor_works': 'PROVIDE_INITIAL_HYPOTHESIS',
                                  'sensor_defective': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS', IsolateProblemCheckEffectiveRadius(),
                     transitions={'isolated_problem': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})


def run():
    """
    Runs the state machine.
    """
    sm = VehicleDiagnosisStateMachine()
    outcome = sm.execute()
    print("OUTCOME:", outcome)


if __name__ == '__main__':
    run()
