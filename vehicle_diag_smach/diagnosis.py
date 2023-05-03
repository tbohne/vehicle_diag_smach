#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from pathlib import Path

import numpy as np
import smach

from obd_ontology import expert_knowledge_enhancer
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from tensorflow import keras
from termcolor import colored

from config import OSCI_SESSION_FILES, TRAINED_MODEL_POOL, Z_NORMALIZATION, \
    SUGGESTION_SESSION_FILE, SESSION_DIR
from low_level_states.select_best_unused_error_code_instance import SelectBestUnusedErrorCodeInstance
from low_level_states.isolate_problem_check_effective_radius import IsolateProblemCheckEffectiveRadius
from low_level_states.no_problem_detected_check_sensor import NoProblemDetectedCheckSensor
from low_level_states.gen_artificial_instance_based_on_cc import GenArtificialInstanceBasedOnCC
from low_level_states.provide_initial_hypothesis_and_log_context import ProvideInitialHypothesisAndLogContext
from low_level_states.suggest_suspect_components import SuggestSuspectComponents
from low_level_states.perform_synchronized_sensor_recordings import PerformSynchronizedSensorRecordings
from low_level_states.perform_data_management import PerformDataManagement


class ProvideDiagAndShowTrace(smach.State):
    """
    State in the low-level SMACH that represents situations in which the diagnosis is provided in combination with
    a detailed trace of all the relevant information that lead to it. Additionally, the diagnosis is uploaded to the
    server.
    """

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['uploaded_diag'],
                             input_keys=['diagnosis'],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_DIAG_AND_SHOW_TRACE' state.

        :param userdata: input of state
        :return: outcome of the state ("uploaded_diag")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PROVIDE_DIAG_AND_SHOW_TRACE", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        # TODO: upload diagnosis to server
        #   - it's important to log the whole context - everything that could be meaningful
        #   - in the long run, this is where we collect the data that we initially lacked, e.g., for automated
        #     data-driven RCA
        #   - to be logged (diagnosis together with):
        #       - associated symptoms, DTCs, components (distinguishing root causes and side effects) etc.

        for key in userdata.diagnosis.keys():
            print("\nidentified anomalous component:", key)
            print("fault path:")
            path = userdata.diagnosis[key][::-1]
            path = [path[i] if i == len(path) - 1 else path[i] + " -> " for i in range(len(path))]
            print(colored("".join(path), "red", "on_white", ["bold"]))

        # TODO: show diagnosis + trace
        return "uploaded_diag"


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
        print("executing", colored("CLASSIFY_OSCILLOGRAMS", "yellow", "on_grey", ["bold"]),
              "state (applying trained model)..")
        print("############################################")

        anomalous_components = []
        non_anomalous_components = []
        num_of_recordings = len(list(Path(SESSION_DIR + "/" + OSCI_SESSION_FILES + "/").rglob('*.csv')))

        # iteratively process oscilloscope recordings
        for osci_path in Path(SESSION_DIR + "/" + OSCI_SESSION_FILES + "/").rglob('*.csv'):
            label = str(osci_path).split("/")[2].replace(".csv", "")
            comp_name = label.split("_")[-1]

            print(colored("\n\nclassifying:" + comp_name, "green", "on_grey", ["bold"]))
            _, voltages = preprocess.read_oscilloscope_recording(osci_path)

            if Z_NORMALIZATION:
                voltages = preprocess.z_normalize_time_series(voltages)

            try:
                # selecting trained model based on component name
                trained_model_file = TRAINED_MODEL_POOL + comp_name + ".h5"
                print("loading trained model:", trained_model_file)
                model = keras.models.load_model(trained_model_file)
            except OSError as e:
                print("no trained model available for the signal (component) to be classified:", comp_name)
                print("adding it to the list of components to be verified manually..")
                userdata.suggestion_list[comp_name] = False
                continue

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
                print(colored("--> ANOMALY DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
                print("#####################################")
                anomalous_components.append(comp_name)
            else:
                print("#####################################")
                print(colored("--> NO ANOMALIES DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
                print("#####################################")
                non_anomalous_components.append(comp_name)

            heatmaps = {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
                        "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
                        "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
                        "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}

            res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"

            # read suggestion - assumption: it is always the latest suggestion
            with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
                suggestions = json.load(f)
            assert comp_name in list(suggestions.values())[0]
            assert len(suggestions.keys()) == 1
            dtc = list(suggestions.keys())[0]
            print("DTC to set heatmap for:", dtc)
            print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
            # extend KG with generated heatmap
            knowledge_enhancer = expert_knowledge_enhancer.ExpertKnowledgeEnhancer("")
            # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
            knowledge_enhancer.extend_kg_with_heatmap_facts(dtc, comp_name, heatmaps["tf-keras-gradcam"].tolist())

            cam.plot_heatmaps_as_overlay(heatmaps, voltages, label + res_str)

        # classifying the subset of components that are to be classified manually
        for comp in userdata.suggestion_list.keys():
            if not userdata.suggestion_list[comp]:
                print(colored("\n\nmanual inspection of component " + comp, "green", "on_grey", ["bold"]))
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


class DiagnosisStateMachine(smach.StateMachine):
    """
    Low-level diagnosis state machine responsible for the details of the diagnostic process.
    """

    def __init__(self):
        super(DiagnosisStateMachine, self).__init__(
            outcomes=['refuted_hypothesis', 'diag'],
            input_keys=[],
            output_keys=[]
        )

        self.userdata.sm_input = []

        # defines states and transitions of the low-level diagnosis SMACH
        with self:
            self.add('SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE', SelectBestUnusedErrorCodeInstance(),
                     transitions={'selected_matching_instance(OBD_CC)': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'no_matching_selected_best_instance': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'no_instance': 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC',
                                  'no_instance_and_CC_already_used': 'NO_PROBLEM_DETECTED_CHECK_SENSOR'},
                     remapping={'selected_instance': 'sm_input',
                                'customer_complaints': 'sm_input'})

            self.add('ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS', IsolateProblemCheckEffectiveRadius(),
                     transitions={'isolated_problem': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={'classified_components': 'sm_input',
                                'fault_paths': 'sm_input'})

            self.add('NO_PROBLEM_DETECTED_CHECK_SENSOR', NoProblemDetectedCheckSensor(),
                     transitions={'sensor_works': 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT',
                                  'sensor_defective': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC', GenArtificialInstanceBasedOnCC(),
                     transitions={'generated_artificial_instance': 'SUGGEST_SUSPECT_COMPONENTS'},
                     remapping={'customer_complaints': 'sm_input',
                                'generated_instance': 'sm_input'})

            self.add('PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT', ProvideInitialHypothesisAndLogContext(),
                     transitions={'no_diag': 'refuted_hypothesis'},
                     remapping={})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(),
                     transitions={'uploaded_diag': 'diag'},
                     remapping={'diagnosis': 'sm_input'})

            self.add('CLASSIFY_OSCILLOGRAMS', ClassifyOscillograms(),
                     transitions={'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE',
                                  'no_anomaly': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'suggestion_list': 'sm_input',
                                'classified_components': 'sm_input'})

            self.add('INSPECT_COMPONENTS', InspectComponents(),
                     transitions={'no_anomaly': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS',
                                  'no_anomaly_and_no_more_measuring_pos': 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE'},
                     remapping={'suggestion_list': 'sm_input'})

            self.add('PERFORM_DATA_MANAGEMENT', PerformDataManagement(),
                     transitions={'performed_data_management': 'CLASSIFY_OSCILLOGRAMS',
                                  'performed_reduced_data_management': 'INSPECT_COMPONENTS'},
                     remapping={'suggestion_list': 'sm_input'})

            self.add('PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS', PerformSynchronizedSensorRecordings(),
                     transitions={'processed_sync_sensor_data': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'suggestion_list': 'sm_input'})

            self.add('SUGGEST_SUSPECT_COMPONENTS', SuggestSuspectComponents(),
                     transitions={'provided_suggestions': 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS',
                                  'no_oscilloscope_required': 'PERFORM_DATA_MANAGEMENT'},
                     remapping={'selected_instance': 'sm_input',
                                'generated_instance': 'sm_input',
                                'suggestion_list': 'sm_input'})
