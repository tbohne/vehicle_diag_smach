#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import logging
import os
import shutil
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import smach
import tensorflow as tf
from OBDOntology import ontology_instance_generator, knowledge_graph_query_tool
from bs4 import BeautifulSoup
from matplotlib.lines import Line2D
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from py4j.java_gateway import JavaGateway
from tensorflow import keras
from termcolor import colored
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
        print("\n\n############################################")
        print("executing", colored("REC_VEHICLE_AND_PROC_USER_DATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # TODO: read from updated GUI
        # GUI.run_gui()
        user_data = {
            "workshop_name": "workshop_one",
            "zipcode": "12345",
            "workshop_id": "00000",
            "mechanic_id": "99999",
            # how many parallel measurements are possible at most (based on workshop equipment / oscilloscope channels)
            "max_number_of_parallel_recordings": "4",
            "date": date.today()
        }

        for k in user_data.keys():
            print(k + ": " + str(user_data[k]))

        # if not present, create directory for session data
        if not os.path.exists(config.SESSION_DIR):
            print("------ creating session data directory..")
            os.makedirs(config.SESSION_DIR + "/")
        else:
            # if it already exists, clear outdated session data
            print("------ clearing session data directory..")
            shutil.rmtree(config.SESSION_DIR)
            os.makedirs(config.SESSION_DIR + "/")

        # write user data to session directory
        with open(config.SESSION_DIR + '/user_data.json', 'w') as f:
            print("------ writing user data to session directory..")
            json.dump(user_data, f, default=str)

        val = None
        while val != "":
            val = input("\n..............................")

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
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROC_CUSTOMER_COMPLAINTS", "yellow", "on_grey", ["bold"]), "state..")
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

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ESTABLISH_INITIAL_HYPOTHESIS' state.

        :param userdata: input of state
        :return: outcome of the state ("established_init_hypothesis" | "no_OBD_and_no_CC")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("ESTABLISH_INITIAL_HYPOTHESIS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        print("reading customer complaints session protocol..")
        initial_hypothesis = ""
        try:
            with open(config.SESSION_DIR + "/" + config.XPS_SESSION_FILE) as f:
                data = f.read()
                session_data = BeautifulSoup(data, 'xml')
                for tag in session_data.find_all('rating', {'type': 'heuristic'}):
                    initial_hypothesis = tag.parent['objectName']
        except FileNotFoundError:
            print("no customer complaints available..")

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

        # TODO: use historical data to refine initial hypothesis (e.g. to deny certain hypotheses)
        print("establish hypothesis..")

        val = None
        while val != "":
            val = input("\n..............................")

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

        for k in obd_data.keys():
            print(k + ": " + str(obd_data[k]))

        return obd_data

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_OBD_data" | "no_OBD_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("READ_OBD_DATA_AND_GEN_ONTOLOGY_INSTANCES", "yellow", "on_grey", ["bold"]), "state..")
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

        val = None
        while val != "":
            val = input("\n..............................")

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

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'RETRIEVE_HISTORICAL_DATA' state.

        Two kinds of information:
        - historical info for specific vehicle (via VIN)
        - historical info for vehicle type (via model)

        :param userdata: input of state
        :return: outcome of the state ("processed_all_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("RETRIEVE_HISTORICAL_DATA", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)
        vin = userdata.vehicle_specific_instance_data_in['vin']
        model = userdata.vehicle_specific_instance_data_in['model']

        # TODO: potentially retrieve more historical information (not only DTCs)
        historic_dtcs_by_vin = qt.query_dtcs_by_vin(vin)
        print("DTCs previously recorded in present car:", historic_dtcs_by_vin)
        print("model to retrieve historical data for:", model)
        historic_dtcs_by_model = qt.query_dtcs_by_model(model)
        print("DTCs previously recorded in model of present car:", historic_dtcs_by_model)

        with open(config.SESSION_DIR + "/" + config.HISTORICAL_INFO_FILE, "w") as f:
            f.write("DTCs previously recorded in car with VIN " + vin + ": " + str(historic_dtcs_by_vin) + "\n")
            f.write("DTCs previously recorded in cars of model " + model + ": " + str(historic_dtcs_by_model) + "\n")

        userdata.vehicle_specific_instance_data_out = userdata.vehicle_specific_instance_data_in

        val = None
        while val != "":
            val = input("\n..............................")

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

    @staticmethod
    def manual_transition() -> None:
        val = None
        while val != "":
            val = input("\n..............................")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SUGGEST_MEASURING_POS_OR_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_suggestions" | "no_oscilloscope_required")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SUGGEST_MEASURING_POS_OR_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # print("generated instance:", userdata.generated_instance)
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)

        # should not be queried over and over again - just once for a session
        # -> then suggest as many as possible per execution of the state (write to session files)
        if not os.path.exists(config.SESSION_DIR + "/" + config.SUS_COMP_TMP_FILE):
            suspect_components = qt.query_suspect_component_by_dtc(userdata.selected_instance)
            # write suspect components to session file
            with open(config.SESSION_DIR + "/" + config.SUS_COMP_TMP_FILE, 'w') as f:
                json.dump(suspect_components, f, default=str)
        else:
            # read remaining suspect components from file
            with open(config.SESSION_DIR + "/" + config.SUS_COMP_TMP_FILE) as f:
                suspect_components = json.load(f)

        print("SUSPECT COMPONENTS:", suspect_components)

        # decide whether oscilloscope required
        oscilloscope_usage = []
        for comp in suspect_components:
            use = qt.query_oscilloscope_usage_by_suspect_component(comp)[0]
            print("comp:", comp, "use oscilloscope:", use)
            oscilloscope_usage.append(use)

        suggestion_list = {comp: osci for comp, osci in zip(suspect_components, oscilloscope_usage)}
        userdata.suggestion_list = suggestion_list

        # everything that is used here should be removed from the tmp file
        with open(config.SESSION_DIR + "/" + config.SUS_COMP_TMP_FILE, 'w') as f:
            json.dump([c for c in suspect_components if c not in suggestion_list.keys()], f, default=str)

        if True in oscilloscope_usage:
            print("\n--> there is at least one suspect component that can be diagnosed using an oscilloscope..")
            self.manual_transition()
            return "provided_suggestions"

        print("none of the identified suspect components can be diagnosed with an oscilloscope..")
        self.manual_transition()
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
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS", "yellow", "on_grey", ["bold"]), "state..")
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

        val = None
        while val != "":
            val = input("\npress 'ENTER' when the recording phase is finished and the oscillograms are generated..")

        # creating dummy oscillograms in '/session_files' for each suspect component
        comp_idx = 0
        for path in Path(config.DUMMY_OSCILLOGRAMS).rglob('*.csv'):
            src = str(path)
            osci_session_dir = config.SESSION_DIR + "/" + config.OSCI_SESSION_FILES + "/"

            if not os.path.exists(osci_session_dir):
                os.makedirs(osci_session_dir)

            shutil.copy(src, osci_session_dir + str(src.split("/")[-1]))
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
                             input_keys=['suggestion_list'],
                             output_keys=['suggestion_list'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PERFORM_DATA_MANAGEMENT' state.

        :param userdata: input of state
        :return: outcome of the state ("performed_data_management" | "performed_reduced_data_management")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PERFORM_DATA_MANAGEMENT", "yellow", "on_grey", ["bold"]), "state..")
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

            val = None
            while val != "":
                val = input("\n..............................")

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
                             outcomes=['no_anomaly', 'detected_anomalies', 'no_anomaly_and_no_more_measuring_pos'],
                             input_keys=['suggestion_list'],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'INSPECT_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("no_anomaly" | "detected_anomalies" | "no_anomaly_and_no_more_measuring_pos")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("INSPECT_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        print("SUGGESTION LIST:", userdata.suggestion_list)
        # TODO: to be implemented
        no_anomaly = True

        # TODO: are there remaining suspect components? (atm every component is suggested each case)
        no_more_measuring_pos = True

        if no_anomaly and no_more_measuring_pos:
            return "no_anomaly_and_no_more_measuring_pos"
        elif no_anomaly:
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
                             input_keys=['suggestion_list'],
                             output_keys=['classified_components'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'CLASSIFY_OSCILLOGRAMS' state.

        :param userdata: input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_and_no_more_measuring_pos")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("CLASSIFY_OSCILLOGRAMS", "yellow", "on_grey", ["bold"]), "state (applying trained model)..")
        print("############################################")
        model = keras.models.load_model(config.TRAINED_MODEL)
        anomalous_components = []
        non_anomalous_components = []
        num_of_recordings = len(list(Path(config.SESSION_DIR + "/" + config.OSCI_SESSION_FILES + "/").rglob('*.csv')))

        # iteratively process oscilloscope recordings
        for osci_path in Path(config.SESSION_DIR + "/" + config.OSCI_SESSION_FILES + "/").rglob('*.csv'):
            label = str(osci_path).split("/")[2].replace(".csv", "")
            comp_name = label.split("_")[-1]
            print("\n\nclassifying:", comp_name)
            _, voltages = preprocess.read_oscilloscope_recording(osci_path)
            voltages = preprocess.z_normalize_time_series(voltages)

            # fix input size
            net_input_size = model.layers[0].output_shape[0][1]

            assert net_input_size == len(voltages)
            # if len(voltages) > net_input_size:
            #     remove = len(voltages) - net_input_size
            #     voltages = voltages[: len(voltages) - remove]

            net_input = np.asarray(voltages).astype('float32')
            net_input = net_input.reshape((net_input.shape[0], 1))

            prediction = model.predict(np.array([net_input]))
            num_classes = len(prediction[0])

            # addresses both models with one output neuron and those with several
            anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5
            pred_value = prediction.max() if num_classes > 1 else prediction[0][0]

            if anomaly:
                print("#####################################")
                print("--> ANOMALY DETECTED (", str(pred_value), ")")
                print("#####################################")
                anomalous_components.append(comp_name)
            else:
                print("#####################################")
                print("--> NO ANOMALIES DETECTED (", str(pred_value), ")")
                print("#####################################")
                non_anomalous_components.append(comp_name)

            heatmaps = {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
                        "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
                        "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
                        "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}

            cam.plot_heatmaps_as_overlay(heatmaps, voltages, label)

        # classifying the subset of components that are to be classified manually
        for comp in userdata.suggestion_list.keys():
            if not userdata.suggestion_list[comp]:
                print("\n\nmanual inspection of component", comp)
                val = ""
                while val not in ['0', '1']:
                    val = input("\npress '0' for defective component, i.e., anomaly, and '1' for no defect..")
                anomaly = val == "0"
                if anomaly:
                    anomalous_components.append(comp)
                else:
                    non_anomalous_components.append(comp)

        classified_components = {}
        for comp in non_anomalous_components:
            classified_components[comp] = False
        for comp in anomalous_components:
            classified_components[comp] = True

        userdata.classified_components = classified_components

        # there are three options:
        #   1. there's only one recording at a time and thus only one classification
        #   2. there are as many parallel recordings as there are suspect components for the DTC
        #   3. there are multiple parallel recordings, but not as many as there are suspect components for the DTC

        # TODO: are there remaining suspect components? (atm every component is suggested each case)
        remaining_suspect_components = False

        if len(anomalous_components) == 0 and not remaining_suspect_components:
            return "no_anomaly_and_no_more_measuring_pos"
        elif len(anomalous_components) == 0 and remaining_suspect_components:
            return "no_anomaly"
        elif len(anomalous_components) > 0:
            return "detected_anomalies"


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

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_diag_and_explanation")
        """
        print("\n\n############################################")
        print("executing", colored("PROVIDE_DIAG_AND_SHOW_TRACE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        for key in userdata.diagnosis.keys():
            print("identified anomalous component:", key)
            print("fault path:")
            path = userdata.diagnosis[key][::-1]
            path = [path[i] if i == len(path) - 1 else path[i] + " -> " for i in range(len(path))]
            print(colored("".join(path), "red", "on_white", ["bold"]))

        # TODO: show diagnosis + trace
        return "provided_diag_and_explanation"


class ProvideInitialHypothesisAndLogContext(smach.State):
    """
    State in the high-level SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['no_diag'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT' state.

        :param userdata: input of state
        :return: outcome of the state ("no_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")
        # TODO: create log file for the failed diagnostic process to improve future diagnosis (missing knowledge etc.)
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

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'UPLOAD_DIAGNOSIS' state.

        :param userdata: input of state
        :return: outcome of the state ("uploaded_diag")
        """
        print("\n\n############################################")
        print("executing", colored("UPLOAD_DIAGNOSIS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")
        # TODO: upload diagnosis to server
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
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC", "yellow", "on_grey", ["bold"]), "state..")
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

    @staticmethod
    def manual_transition() -> None:
        val = None
        while val != "":
            val = input("\n..............................")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SELECT_BEST_UNUSED_DTC_INSTANCE' state.

        :param userdata: input of state
        :return: outcome of the state ("selected_matching_instance(OBD_CC)" | "no_matching_selected_best_instance" |
                                       "no_instance" | "no_instance_and_CC_already_used")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("SELECT_BEST_UNUSED_DTC_INSTANCE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # load DTC instances from tmp file
        with open(config.SESSION_DIR + "/" + config.DTC_TMP_FILE) as f:
            dtc_list = json.load(f)['list']

        customer_complaints_list = []
        try:
            # load customer complaints from tmp file
            with open(config.SESSION_DIR + "/" + config.CC_TMP_FILE) as f:
                customer_complaints_list = json.load(f)['list']
        except FileNotFoundError:
            pass

        # case 1: no DTC instance provided, but CC still available
        if len(dtc_list) == 0 and len(customer_complaints_list) == 1:
            # this option leads to the customer complaints being used to generate an artificial DTC instance
            self.remove_cc_instance_from_tmp_file()
            userdata.customer_complaints = customer_complaints_list[0]
            print("no DTCs provided, but customer complaints available..")
            self.manual_transition()
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
                    print("select matching instance (OBD, CC)..")
                    self.manual_transition()
                    return "selected_matching_instance(OBD_CC)"
            # sub-case 2: no matching instance -> select best instance
            # TODO: select best remaining DTC instance based on some criteria
            userdata.selected_instance = dtc_list[0]
            dtc_list.remove(dtc_list[0])
            self.remove_dtc_instance_from_tmp_file(dtc_list)
            print("DTCs and customer complaints available, but no matching instance..")
            self.manual_transition()
            return "no_matching_selected_best_instance"

        # case 3: no remaining instance and customer complaints already used
        elif len(dtc_list) == 0 and len(customer_complaints_list) == 0:
            print("no more DTC instances and customer complaints already considered..")
            self.manual_transition()
            return "no_instance_and_CC_already_used"

        # case 4: no customer complaints, but remaining DTCs
        else:
            # TODO: select best remaining DTC instance based on some criteria
            selected_dtc = dtc_list[0]
            userdata.selected_instance = selected_dtc
            dtc_list.remove(selected_dtc)
            self.remove_dtc_instance_from_tmp_file(dtc_list)
            print("no customer complaints available, selecting DTC instance..")
            print("selected DTC instance:", selected_dtc)
            self.manual_transition()
            return "no_matching_selected_best_instance"


class NoProblemDetectedCheckSensor(smach.State):
    """
    State in the high-level SMACH that represents situations in which no actual anomaly was detected and the indirect
    conclusion of a potential sensor malfunction is provided. This conclusion should be verified / refuted in this
    state.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['sensor_works', 'sensor_defective'],
                             input_keys=[''],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'NO_PROBLEM_DETECTED_CHECK_SENSOR' state.

        :param userdata: input of state
        :return: outcome of the state ("sensor_works" | "sensor_defective")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("NO_PROBLEM_DETECTED_CHECK_SENSOR", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        print("no anomaly identified -- check potential sensor malfunction..")

        val = ""
        while val not in ['0', '1']:
            val = input("\npress '0' for sensor malfunction and '1' for working sensor..")

        if val == "0":
            return "sensor_defective"
        return "sensor_works"


class IsolateProblemCheckEffectiveRadius(smach.State):
    """
    State in the high-level SMACH that represents situations in which one or more anomalies have been detected, and the
    task is to isolate the defective components based on their effective radius (structural knowledge).
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['isolated_problem'],
                             input_keys=['classified_components'],
                             output_keys=['fault_paths'])

        self.model = keras.models.load_model(config.TRAINED_MODEL)
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)

    def classify_component(self, affecting_comp: str) -> bool:
        """
        Classifies the oscillogram for the specified vehicle component.

        :param affecting_comp: component to classify oscillogram for
        :return: whether an anomaly has been detected
        """
        # create session data directory
        osci_iso_session_dir = config.SESSION_DIR + "/" + config.OSCI_ISOLATION_SESSION_FILES + "/"
        if not os.path.exists(osci_iso_session_dir):
            os.makedirs(osci_iso_session_dir)

        val = None
        while val != "":
            val = input("\npress 'ENTER' when the recording phase is finished and the" +
                        " oscillogram is generated..")

        # TODO: hard-coded for demo purposes - showing reasonable case (one NEG)
        if affecting_comp in ["Ladedruck-Regelventil", "Ladedrucksteller-Positionssensor"]:
            shutil.copy(config.DUMMY_ISOLATION_OSCILLOGRAM_NEG, osci_iso_session_dir + affecting_comp + ".csv")
        else:
            shutil.copy(config.DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_iso_session_dir + affecting_comp + ".csv")
        path = config.SESSION_DIR + "/" + config.OSCI_ISOLATION_SESSION_FILES + "/" + affecting_comp + ".csv"
        _, voltages = preprocess.read_oscilloscope_recording(path)
        voltages = preprocess.z_normalize_time_series(voltages)

        net_input_size = self.model.layers[0].output_shape[0][1]
        assert net_input_size == len(voltages)
        # if len(voltages) > net_input_size:
        #     remove = len(voltages) - net_input_size
        #     voltages = voltages[: len(voltages) - remove]

        net_input = np.asarray(voltages).astype('float32')
        net_input = net_input.reshape((net_input.shape[0], 1))

        prediction = self.model.predict(np.array([net_input]))
        num_classes = len(prediction[0])
        # addresses both models with one output neuron and those with several
        anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5

        if anomaly:
            print("#####################################")
            print("--> ANOMALY DETECTED")
            print("#####################################")
        else:
            print("#####################################")
            print("--> NO ANOMALIES DETECTED")
            print("#####################################")

        heatmaps = {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), self.model, prediction),
                    "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), self.model, prediction),
                    "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), self.model, prediction),
                    "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), self.model, prediction)}

        cam.plot_heatmaps_as_overlay(heatmaps, voltages, path.split("/")[2].replace(".csv", ""))

        return np.argmax(prediction) == 0

    def construct_complete_graph(self, graph: dict, components_to_process: list) -> dict:
        """
        Recursive function that constructs the complete causal graph for the specified components.

        :param graph: partial graph to be extended
        :param components_to_process: components yet to be processed
        :return: constructed causal graph
        """
        if len(components_to_process) == 0:
            return graph

        comp = components_to_process.pop(0)
        if comp not in graph.keys():
            affecting_comp = self.qt.query_affected_by_relations_by_suspect_component(comp, False)
            components_to_process += affecting_comp
            graph[comp] = affecting_comp
        return self.construct_complete_graph(graph, components_to_process)

    @staticmethod
    def create_legend_lines(colors, **kwargs):
        """
        Creates the edge representations for the plot legend.

        :param colors: colors for legend lines
        :return: generated line representations
        """
        return Line2D([0, 1], [0, 1], color=colors, **kwargs)

    def visualize_causal_graphs(self, anomalous_paths, complete_graphs, explicitly_considered_links):
        """
        Visualizes the causal graphs along with the actual paths to the root cause.

        :param anomalous_paths: the paths to the root cause
        :param complete_graphs: the causal graphs
        :param explicitly_considered_links: links that have been verified explicitly
        """
        print("isolation results, i.e., causal paths:")
        for key in anomalous_paths.keys():
            print(key, ":", anomalous_paths[key])

        for key in complete_graphs.keys():
            print("visualizing graph for component:", key)
            plt.title("Causal Graph (Network of Effective Connections) for " + key, fontsize=24, fontweight='bold')

            from_relations = [k for k in complete_graphs[key].keys() for _ in range(len(complete_graphs[key][k]))]
            to_relations = [complete_graphs[key][k] for k in complete_graphs[key].keys()]
            to_relations = [item for lst in to_relations for item in lst]

            causal_links = []
            for i in range(len(to_relations)):
                for j in range(len(anomalous_paths[key]) - 1):
                    # causal link check
                    if anomalous_paths[key][j] == from_relations[i] and anomalous_paths[key][j + 1] == to_relations[i]:
                        causal_links.append(i)
                        break

            colors = ['g' if i not in causal_links else 'r' for i in range(len(to_relations))]
            for i in range(len(from_relations)):
                # if the from-to relation is not part of the actually considered links, it should be black
                if to_relations[i] not in explicitly_considered_links[from_relations[i]]:
                    colors[i] = 'black'

            widths = [5 if i not in causal_links else 10 for i in range(len(to_relations))]
            df = pd.DataFrame({'from': from_relations, 'to': to_relations})

            g = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
            pos = nx.spring_layout(g, seed=68)
            nx.draw(g, pos=pos, with_labels=True, node_size=40000, alpha=0.75, arrows=True, edge_color=colors, width=widths)

            legend_lines = [self.create_legend_lines(clr, lw=5) for clr in ['r', 'g', 'black']]
            labels = ["fault path", "non-anomalous links", "disregarded"]
            plt.legend(legend_lines, labels, fontsize=18)
            plt.show()

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS' state.
        Implements the search in the causal graph (effect network).

        :param userdata: input of state
        :return: outcome of the state ("isolated_problem")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # already checked components together with the corresponding results (true -> anomaly)
        already_checked_components = userdata.classified_components.copy()
        anomalous_paths = {}
        print("constructing causal graph, i.e., subgraph of structural component knowledge..")
        complete_graphs = {comp: self.construct_complete_graph({}, [comp])
                           for comp in userdata.classified_components.keys() if userdata.classified_components[comp]}
        explicitly_considered_links = {}

        for anomalous_comp in userdata.classified_components.keys():
            if not userdata.classified_components[anomalous_comp]:
                continue
            print("isolating", anomalous_comp, "..")
            affecting_components = self.qt.query_affected_by_relations_by_suspect_component(anomalous_comp)

            if anomalous_comp not in list(explicitly_considered_links.keys()):
                explicitly_considered_links[anomalous_comp] = affecting_components.copy()
            else:
                explicitly_considered_links[anomalous_comp] += affecting_components.copy()

            print("component potentially affected by:", affecting_components)
            unisolated_anomalous_components = affecting_components
            causal_path = [anomalous_comp]

            while len(unisolated_anomalous_components) > 0:
                comp_to_be_checked = unisolated_anomalous_components.pop(0)
                print("component to be checked:", comp_to_be_checked)
                if comp_to_be_checked not in list(explicitly_considered_links.keys()):
                    explicitly_considered_links[comp_to_be_checked] = []

                if comp_to_be_checked in already_checked_components.keys():
                    print("already checked this component - anomaly:", already_checked_components[comp_to_be_checked])
                    if already_checked_components[comp_to_be_checked]:
                        causal_path.append(comp_to_be_checked)
                        affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(comp_to_be_checked)
                        unisolated_anomalous_components += affecting_comps
                        explicitly_considered_links[comp_to_be_checked] += affecting_comps.copy()
                    continue

                use_oscilloscope = self.qt.query_oscilloscope_usage_by_suspect_component(comp_to_be_checked)[0]

                if use_oscilloscope:
                    print("use oscilloscope..")
                    anomaly = self.classify_component(comp_to_be_checked)
                    already_checked_components[comp_to_be_checked] = anomaly
                else:
                    print("manual inspection of component (no oscilloscope)..")
                    val = ""
                    while val not in ['0', '1']:
                        val = input("\npress '0' for defective component, i.e., anomaly, and '1' for no defect..")
                    anomaly = val == "0"
                    already_checked_components[comp_to_be_checked] = anomaly

                if anomaly:
                    causal_path.append(comp_to_be_checked)
                    affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(comp_to_be_checked)
                    unisolated_anomalous_components += affecting_comps
                    explicitly_considered_links[comp_to_be_checked] += affecting_comps.copy()

            anomalous_paths[anomalous_comp] = causal_path

        self.visualize_causal_graphs(anomalous_paths, complete_graphs, explicitly_considered_links)
        userdata.fault_paths = anomalous_paths
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
                     remapping={'suggestion_list': 'sm_input'})

            self.add('INSPECT_COMPONENTS', InspectComponents(),
                     transitions={'no_anomaly': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS',
                                  'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_DTC_INSTANCE'},
                     remapping={'suggestion_list': 'sm_input'})

            self.add('CLASSIFY_OSCILLOGRAMS', ClassifyOscillograms(),
                     transitions={'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_DTC_INSTANCE',
                                  'no_anomaly': 'SUGGEST_MEASURING_POS_OR_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'suggestion_list': 'sm_input',
                                'classified_components': 'sm_input'})

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
                     remapping={'classified_components': 'sm_input',
                                'fault_paths': 'sm_input'})


def run():
    """
    Runs the state machine.
    """
    sm = VehicleDiagnosisStateMachine()
    tf.get_logger().setLevel(logging.ERROR)
    outcome = sm.execute()
    # print("OUTCOME:", outcome)


if __name__ == '__main__':
    run()
