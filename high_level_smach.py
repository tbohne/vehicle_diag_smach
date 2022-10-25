#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import smach
from OBDOntology import ontology_instance_generator, knowledge_graph_query_tool
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
        smach.State.__init__(self, outcomes=['processed_user_data'], input_keys=[''], output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'REC_VEHICLE_AND_PROC_USER_DATA' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_user_data")
        """
        print("############################################")
        print("executing REC_VEHICLE_AND_PROC_USER_DATA state..")
        print("############################################")

        # TODO: read from updated GUI
        # GUI.run_gui()
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
        smach.State.__init__(self, outcomes=['received_complaints', 'no_complaints'], input_keys=[''], output_keys=[''])

    @staticmethod
    def launch_customer_xps() -> str:
        """
        Launches the expert system that processes the customer complaints.
        """
        print("establish connection to customer XPS server..")
        gateway = JavaGateway()
        customer_xps = gateway.entry_point
        return customer_xps.demo("../" + config.SESSION_DIR + "/" + config.XPS_SESSION_FILE)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROC_CUSTOMER_COMPLAINTS' state.

        :param userdata: input of state (provided user data)
        :return: outcome of the state ("received_complaints" | "no_complaints")
        """
        print("############################################")
        print("executing PROC_CUSTOMER_COMPLAINTS state..")
        print("############################################")
        val = ""
        while val != "0" and val != "1":
            val = input("starting diagnosis with [0] / without [1] customer complaints")

        if val == "0":
            print("result of customer xps: ", self.launch_customer_xps())
            print("customer XPS session protocol saved..")
            return "received_complaints"
        else:
            print("starting diagnosis without customer complaints..")
            return "no_complaints"


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

    def execute(self, userdata):
        """
        Execution of 'ESTABLISH_INITIAL_HYPOTHESIS' state.

        :param userdata: input of state
        :return: outcome of the state ("established_init_hypothesis" | "no_OBD_and_no_CC")
        """
        print("############################################")
        print("executing ESTABLISH_INITIAL_HYPOTHESIS state..")
        print("############################################")

        print("reading customer complaints session protocol..")
        initial_hypothesis = ""
        with open(config.SESSION_DIR + "/" + config.XPS_SESSION_FILE) as f:
            data = f.read()
            session_data = BeautifulSoup(data, 'xml')
            for tag in session_data.find_all('rating', {'type': 'heuristic'}):
                initial_hypothesis = tag.parent['objectName']

        if len(userdata.vehicle_specific_instance_data['dtc_list']) == 0 and len(initial_hypothesis) == 0:
            # no OBD data + no customer complaints -> insufficient data
            return "no_OBD_and_no_CC"

        print("reading historical information..")
        with open(config.SESSION_DIR + "/" + config.HISTORICAL_INFO_FILE) as f:
            data = f.read()

        if len(initial_hypothesis) > 0:
            print("initial hypothesis based on customer complaints available..")
            print("initial hypothesis:", initial_hypothesis)
            userdata.hypothesis = initial_hypothesis
            unused_cc = {'list': [initial_hypothesis]}
            with open(config.SESSION_DIR + "/" + config.CC_TMP_FILE, 'w') as f:
                json.dump(unused_cc, f, default=str)
        else:
            print("no initial hypothesis based on customer complaints..")

        # TODO: compare customer complaints with OBD data (matching?)
        # TODO: use historical data to refine initial hypothesis (e.g. to deny certain hypotheses)
        print("establish hypothesis..")
        return "established_init_hypothesis"


class ReadOBDDataAndGenOntologyInstances(smach.State):
    """
    State in the high-level SMACH that represents situations in which the OBD information are read from the ECU.
    Based on the read information, an ontology instance is generated, i.e., the vehicle-specific instance data
    is entered into the knowledge graph.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['processed_OBD_data', 'no_OBD_data'],
                             input_keys=[''],
                             output_keys=['vehicle_specific_instance_data'])

    @staticmethod
    def parse_obd_logfile() -> dict:
        print("processing OBD log file..")
        obd_data = {"dtc_list": [], "model": "", "hsn": "", "tsn": "", "vin": ""}

        with open(config.SAMPLE_OBD_LOG) as f:
            obd_lines = f.readlines()

        # TODO: parse DTCs from OBD log (above)
        obd_data['dtc_list'] = ['P2563']
        obd_data['model'] = "Mazda 3"
        obd_data['hsn'] = "849357984"
        obd_data['tsn'] = "453948539"
        obd_data['vin'] = "1234567890ABCDEFGHJKLMNPRSTUVWXYZ"
        return obd_data

    def execute(self, userdata):
        """
        Execution of 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_OBD_data" | "no_OBD_data")
        """
        print("############################################")
        print("executing READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES state..")
        print("############################################")
        vehicle_specific_instance_data = self.parse_obd_logfile()
        if len(vehicle_specific_instance_data['dtc_list']) == 0:
            return "no_OBD_data"

        # write OBD data to session file
        with open(config.SESSION_DIR + "/" + config.OBD_INFO_FILE, "w") as f:
            json.dump(vehicle_specific_instance_data, f, default=str)
        # also create tmp file for unused DTC instances
        with open(config.SESSION_DIR + "/" + config.DTC_TMP_FILE, "w") as f:
            dtc_tmp = {'list': vehicle_specific_instance_data['dtc_list']}
            json.dump(dtc_tmp, f, default=str)

        # extend knowledge graph with read OBD data (if the vehicle instance already exists, it will be extended)
        instance_gen = ontology_instance_generator.OntologyInstanceGenerator(config.OBD_ONTOLOGY_PATH, local_kb=False)
        for dtc in vehicle_specific_instance_data['dtc_list']:
            instance_gen.extend_knowledge_graph(
                vehicle_specific_instance_data['model'], vehicle_specific_instance_data['hsn'],
                vehicle_specific_instance_data['tsn'], vehicle_specific_instance_data['vin'], dtc
            )
        userdata.vehicle_specific_instance_data = vehicle_specific_instance_data
        return "processed_OBD_data"


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

    def execute(self, userdata):
        """
        Execution of 'RETRIEVE_HISTORICAL_DATA' state.

        Two kinds of information:
        - historical info for specific vehicle (via VIN)
        - historical info for vehicle type (via model)

        :param userdata: input of state
        :return: outcome of the state ("processed_all_data")
        """
        print("############################################")
        print("executing RETRIEVE_HISTORICAL_DATA state..")
        print("############################################")
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)
        vin = userdata.vehicle_specific_instance_data_in['vin']
        model = userdata.vehicle_specific_instance_data_in['model']

        # TODO: potentially retrieve more historical information (not only DTCs)
        print("VIN to retrieve historical data for:", vin)
        historic_dtcs_by_vin = qt.query_dtcs_by_vin(vin)
        print("DTCs previously recorded in present car:", historic_dtcs_by_vin)
        print("model to retrieve historical data for:", model)
        historic_dtcs_by_model = qt.query_dtcs_by_model(model)
        print("DTCs previously recorded in model of present car:", historic_dtcs_by_model)

        with open(config.SESSION_DIR + "/" + config.HISTORICAL_INFO_FILE, "w") as f:
            f.write("DTCs previously recorded in car with VIN " + vin + ": " + str(historic_dtcs_by_vin) + "\n")
            f.write("DTCs previously recorded in cars of model " + model + ": " + str(historic_dtcs_by_model) + "\n")

        userdata.vehicle_specific_instance_data_out = userdata.vehicle_specific_instance_data_in

        return "processed_all_data"


class SuggestMeasuringPosOrComponents(smach.State):
    """
    State in the high-level SMACH that represents situations in which measuring positions or at least suspect
    components in the car are suggested based on the available information (OBD, CC, etc.).
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['provided_suggestions', 'no_oscilloscope_required'],
                             input_keys=['selected_instance', 'generated_instance'],
                             output_keys=['suggestion_list'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SUGGEST_MEASURING_POS_OR_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_suggestions" | "no_oscilloscope_required")
        """
        print("############################################")
        print("executing SUGGEST_MEASURING_POS_OR_COMPONENTS state..")
        print("############################################")

        print("selected instance:", userdata.selected_instance)
        # print("generated instance:", userdata.generated_instance)

        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)
        suspect_components = qt.query_suspect_component_by_dtc(userdata.selected_instance)

        # decide whether oscilloscope required
        oscilloscope_usage = []
        for comp in suspect_components:
            use = qt.query_oscilloscope_usage_by_suspect_component(comp)
            print("comp:", comp, "use oscilloscope:", use)
            oscilloscope_usage.append(use[0])

        userdata.suggestion_list = {comp: osci for comp, osci in zip(suspect_components, oscilloscope_usage)}

        if True in oscilloscope_usage:
            print("there's at least one suspect component that could be diagnosed with an oscilloscope..")
            return "provided_suggestions"

        print("none of the identified suspect components can be diagnosed with an oscilloscope..")
        return "no_oscilloscope_required"


class PerformSynchronizedSensorRecordings(smach.State):
    """
    State in the high-level SMACH that represents situations in which the synchronized sensor recordings are performed
    at the suggested measuring pos / suspect components.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['processed_sync_sensor_data'],
                             input_keys=['suggestion_list'],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_sync_sensor_data")
        """
        print("############################################")
        print("executing PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS state..")
        print("############################################")

        components_to_be_recorded = [k for k, v in userdata.suggestion_list.items() if v]
        components_to_be_manually_verified = [k for k, v in userdata.suggestion_list.items() if not v]
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be manually verified:", components_to_be_manually_verified)
        print("------------------------------------------")

        # TODO: perform manual verification of components and let mechanic enter result + communicate
        #       anomalies further for fault isolation

        print("perform synchronized sensor recordings at:")
        for comp in components_to_be_recorded:
            print("-", comp)

        val = ""
        while val != "0":
            val = input("\npress '0' when the recording phase is finished and the oscillograms are generated..")

        # creating dummy oscillograms in '/session_files' for each suspect component
        comp_idx = 0
        for path in Path(config.DUMMY_OSCILLOGRAMS).rglob('*.csv'):
            src = str(path)
            osci_session_dir = config.SESSION_DIR + "/" + config.OSCI_SESSION_FILES + "/"

            if not os.path.exists(osci_session_dir):
                os.makedirs(osci_session_dir)

            shutil.copy(src, osci_session_dir + components_to_be_recorded[comp_idx] + ".csv")
            comp_idx += 1
            if comp_idx == len(components_to_be_recorded):
                break

        return "processed_sync_sensor_data"


class PerformDataManagement(smach.State):
    """
    State in the high-level SMACH that represents situations in which data management is performed, e.g.:
        - upload all the generated session files (data) to the server
        - retrieve the latest trained classification model from the server
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['performed_data_management', 'performed_reduced_data_management'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PERFORM_DATA_MANAGEMENT' state.

        :param userdata: input of state
        :return: outcome of the state ("performed_data_management" | "performed_reduced_data_management")
        """
        print("############################################")
        print("executing PERFORM_DATA_MANAGEMENT state..")
        print("############################################")

        # TODO: optionally retrieve latest version of trained classifier from server
        print("retrieving latest version of trained classifier from server..")

        # TODO: actually read session data
        print("reading customer complaints from session files..")
        print("reading OBD data from session files..")
        print("reading historical info from session files..")
        print("reading user data from session files..")
        print("reading XPS interview data from session files..")

        # determine whether oscillograms have been generated
        osci_session_dir = config.SESSION_DIR + "/" + config.OSCI_SESSION_FILES + "/"
        if os.path.exists(osci_session_dir):
            print("reading recorded oscillograms from session files..")
            # TODO:
            #   - EDC (Eclipse Dataspace Connector) communication
            #   - consolidate + upload read session data to server
            print("uploading session data to server..")
            return "performed_data_management"

        # TODO:
        #   - EDC (Eclipse Dataspace Connector) communication
        #   - consolidate + upload read session data to server
        print("uploading reduced session data to server..")
        return "performed_reduced_data_management"


class InspectComponents(smach.State):
    """
    State in high-level SMACH representing situations where manual inspection of suspect components for which
    oscilloscope diagnosis is not appropriate is performed.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['no_anomaly', 'detected_anomalies'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata):
        """
        Execution of 'INSPECT_COMPONENTS' state.

        :param userdata:  input of state
        :return: outcome of the state ("no_anomaly" | "detected_anomalies")
        """
        print("############################################")
        print("executing INSPECT_COMPONENTS state..")
        print("############################################")
        # TODO: to be implemented
        if True:
            return "no_anomaly"
        return "detected_anomalies"


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


class ProvideInitialHypothesisAndLogContext(smach.State):
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
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT' state.

        :param userdata:  input of state
        :return: outcome of the state ("no_diag")
        """
        print("############################################")
        print("executing PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT state..")
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
    State in the high-level SMACH that represents situations in which an artificial DTC instance is generated
    based on the customer complaints. Used for cases where no OBD information is available.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['generated_artificial_instance'],
                             input_keys=['customer_complaints'],
                             output_keys=['generated_instance'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC' state.

        :param userdata: input of state
        :return: outcome of the state ("generated_artificial_instance")
        """
        print("############################################")
        print("executing GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC state..")
        print("############################################")
        print("CC:", userdata.customer_complaints)
        # TODO: generate instance based on provided CC
        userdata.generated_instance = "P2563"
        return "generated_artificial_instance"


class SelectBestUnusedErrorCodeInstance(smach.State):
    """
    State in the high-level SMACH that represents situations in which a best-suited, unused DTC instance is
    selected for further processing.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['selected_matching_instance(OBD_CC)', 'no_matching_selected_best_instance',
                                       'no_instance', 'no_instance_and_CC_already_used'],
                             input_keys=[''],
                             output_keys=['selected_instance', 'customer_complaints'])

    @staticmethod
    def remove_dtc_instance_from_tmp_file(remaining_instances: list) -> None:
        """
        Updates the list of unused DTC instances in the corresponding tmp file.

        :param remaining_instances: updated list to save in tmp file
        """
        with open(config.SESSION_DIR + "/" + config.DTC_TMP_FILE, "w") as f:
            json.dump({'list': remaining_instances}, f, default=str)

    @staticmethod
    def remove_cc_instance_from_tmp_file() -> None:
        """
        Clears the customer complaints tmp file.
        """
        with open(config.SESSION_DIR + "/" + config.CC_TMP_FILE, 'w') as f:
            # clear list, already used now
            json.dump({'list': []}, f, default=str)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SELECT_BEST_UNUSED_DTC_INSTANCE' state.

        :param userdata: input of state
        :return: outcome of the state ("selected_matching_instance(OBD_CC)" | "no_matching_selected_best_instance" |
                                       "no_instance" | "no_instance_and_CC_already_used")
        """
        print("############################################")
        print("executing SELECT_BEST_UNUSED_DTC_INSTANCE state..")
        print("############################################")

        # load DTC instances from tmp file
        with open(config.SESSION_DIR + "/" + config.DTC_TMP_FILE) as f:
            dtc_list = json.load(f)['list']

        # load customer complaints from tmp file
        with open(config.SESSION_DIR + "/" + config.CC_TMP_FILE) as f:
            customer_complaints_list = json.load(f)['list']

        # case 1: no DTC instance provided, but CC still available
        if len(dtc_list) == 0 and len(customer_complaints_list) == 1:
            # this option leads to the customer complaints being used to generate an artificial DTC instance
            self.remove_cc_instance_from_tmp_file()
            userdata.customer_complaints = customer_complaints_list[0]
            return "no_instance"

        # case 2: both available
        elif len(dtc_list) > 0 and len(customer_complaints_list) == 1:
            # sub-case 1: matching instance
            for dtc in dtc_list:
                # TODO: check whether DTC matches CC
                match = True
                if match:
                    userdata.selected_instance = dtc
                    dtc_list.remove(dtc)
                    self.remove_dtc_instance_from_tmp_file(dtc_list)
                    return "selected_matching_instance(OBD_CC)"
            # sub-case 2: no matching instance -> select best instance
            # TODO: select best remaining DTC instance based on some criteria
            userdata.selected_instance = dtc_list[0]
            dtc_list.remove(dtc_list[0])
            self.remove_dtc_instance_from_tmp_file(dtc_list)
            return "no_matching_selected_best_instance"

        # case 3: no remaining instance and customer complaints already used
        elif len(dtc_list) == 0 and len(customer_complaints_list) == 0:
            return "no_instance_and_CC_already_used"


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
                                'vehicle_specific_instance_data': 'sm_input'})

            self.add('ESTABLISH_INITIAL_HYPOTHESIS', EstablishInitialHypothesis(),
                     transitions={'established_init_hypothesis': 'SELECT_BEST_UNUSED_DTC_INSTANCE',
                                  'no_OBD_and_no_CC': 'insufficient_data'},
                     remapping={'vehicle_specific_instance_data': 'sm_input',
                                'hypothesis': 'sm_input'})

            self.add('RETRIEVE_HISTORICAL_DATA', RetrieveHistoricalData(),
                     transitions={'processed_all_data': 'ESTABLISH_INITIAL_HYPOTHESIS'},
                     remapping={'vehicle_specific_instance_data_in': 'sm_input',
                                'vehicle_specific_instance_data_out': 'sm_input'})

            self.add('SUGGEST_MEASURING_POS_OR_COMPONENTS', SuggestMeasuringPosOrComponents(),
                     transitions={'provided_suggestions': 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS',
                                  'no_oscilloscope_required': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'selected_instance': 'sm_input',
                                'generated_instance': 'sm_input',
                                'suggestion_list': 'sm_input'})

            self.add('PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS', PerformSynchronizedSensorRecordings(),
                     transitions={'processed_sync_sensor_data': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'suggestion_list': 'sm_input'})

            self.add('PERFORM_DATA_MANAGEMENT', PerformDataManagement(),
                     transitions={'performed_data_management': 'CLASSIFY_OSCILLOGRAMS',
                                  'performed_reduced_data_management': 'INSPECT_COMPONENTS'},
                     remapping={})

            self.add('INSPECT_COMPONENTS', InspectComponents(),
                     transitions={'no_anomaly': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={})

            self.add('CLASSIFY_OSCILLOGRAMS', ClassifyOscillograms(),
                     transitions={'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_DTC_INSTANCE',
                                  'no_anomaly': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'oscillogram': 'sm_input',
                                'diagnosis': 'sm_input'})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(),
                     transitions={'provided_diag_and_explanation': 'UPLOAD_DIAGNOSIS'},
                     remapping={'diagnosis': 'sm_input'})

            self.add('PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT', ProvideInitialHypothesisAndLogContext(),
                     transitions={'no_diag': 'refuted_hypothesis'},
                     remapping={})

            self.add('UPLOAD_DIAGNOSIS', UploadDiagnosis(),
                     transitions={'uploaded_diag': 'diag'},
                     remapping={})

            self.add('GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC', GenArtificialInstanceBasedOnCC(),
                     transitions={'generated_artificial_instance': 'SUGGEST_MEASURING_POS_OR_COMPONENTS'},
                     remapping={'customer_complaints': 'sm_input',
                                'generated_instance': 'sm_input'})

            self.add('SELECT_BEST_UNUSED_DTC_INSTANCE', SelectBestUnusedErrorCodeInstance(),
                     transitions={'selected_matching_instance(OBD_CC)': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'no_matching_selected_best_instance': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'no_instance': 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC',
                                  'no_instance_and_CC_already_used': 'NO_PROBLEM_DETECTED_CHECK_SENSOR'},
                     remapping={'selected_instance': 'sm_input',
                                'customer_complaints': 'sm_input'})

            self.add('NO_PROBLEM_DETECTED_CHECK_SENSOR', NoProblemDetectedCheckSensor(),
                     transitions={'sensor_works': 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT',
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
