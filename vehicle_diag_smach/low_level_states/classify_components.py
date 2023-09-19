#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import smach
from obd_ontology import ontology_instance_generator
from oscillogram_classification import cam
from tensorflow import keras
from termcolor import colored

from vehicle_diag_smach import util
from vehicle_diag_smach.config import SESSION_DIR, SUGGESTION_SESSION_FILE, CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class ClassifyComponents(smach.State):
    """
    State in the low-level SMACH that represents situations in which the suggested physical components in the vehicle
    are classified:
        - synchronized sensor recordings are performed at the suggested suspect components
        - recorded oscillograms are classified using the trained neural net models, i.e., detecting anomalies
        - manual inspection of suspect components, for which oscilloscope diagnosis is not appropriate, is performed
    """

    def __init__(self, model_accessor: ModelAccessor, data_accessor: DataAccessor, data_provider: DataProvider,
                 kg_url: str) -> None:
        """
        Initializes the state.

        :param model_accessor: implementation of the model accessor interface
        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['detected_anomalies', 'no_anomaly', 'no_anomaly_no_more_comp'],
                             input_keys=['suggestion_list'],
                             output_keys=['classified_components'])
        self.model_accessor = model_accessor
        self.data_accessor = data_accessor
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)

    @staticmethod
    def log_classification_actions(
            classified_components: Dict[str, bool], manually_inspected_components: List[str],
            classification_instances: Dict[str, str]
    ) -> None:
        """
        Logs the classification actions to the session directory.

        :param classified_components: dictionary of classified components + classification results
        :param manually_inspected_components: components that were classified manually by the mechanic
        :param classification_instances: IDs of the classification instances by component name
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            for k, v in classified_components.items():
                new_data = {
                    k: v,
                    "State": "CLASSIFY_COMPONENTS",
                    "Classification Type": "manual inspection"
                    if k in manually_inspected_components else "osci classification",
                    "Classification ID": classification_instances[k]
                }
                log_file.extend([new_data])
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("CLASSIFY_COMPONENTS", "yellow", "on_grey", ["bold"]),
              "state (applying trained model)..")
        print("############################################")

    @staticmethod
    def perform_synchronized_sensor_recordings(
            suggestion_list: Dict[str, Tuple[str, bool]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Performs synchronized sensor recordings based on the provided suggestion list.

        :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
        :return: tuple of components to be recorded and components to be verified manually
                 ({comp: reason_for}, {comp: reason_for})
        """
        components_to_be_recorded = {k: v[0] for k, v in suggestion_list.items() if v[1]}
        components_to_be_manually_verified = {k: v[0] for k, v in suggestion_list.items() if not v[1]}
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be verified manually:", components_to_be_manually_verified)
        print("------------------------------------------")
        print(colored("\nperform synchronized sensor recordings at:", "green", "on_grey", ["bold"]))
        for comp in components_to_be_recorded.keys():
            print(colored("- " + comp, "green", "on_grey", ["bold"]))
        return components_to_be_recorded, components_to_be_manually_verified

    @staticmethod
    def log_anomaly(pred_value: float) -> None:
        """
        Logs anomalies.

        :param pred_value: prediction (output value of neural net)
        """
        print("#####################################")
        print(colored("--> ANOMALY DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
        print("#####################################")

    @staticmethod
    def log_regular(pred_value: float) -> None:
        """
        Logs regular cases, i.e., non-anomalies.

        :param pred_value: prediction (output value of neural net)
        """
        print("#####################################")
        print(colored("--> NO ANOMALIES DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
        print("#####################################")

    @staticmethod
    def gen_heatmaps(net_input: np.ndarray, model: keras.models.Model, prediction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates the heatmaps (visual explanations) for the classification.

        :param net_input: input sample
        :param model: trained classification model
        :param prediction: prediction, i.e., outcome of the model
        :return: dictionary of different heatmaps
        """
        return {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
                "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
                "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
                "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}

    @staticmethod
    def log_corresponding_dtc(osci_data: OscillogramData) -> None:
        """
        Logs the corresponding DTC to set the heatmaps for, i.e., reads the DTC suggestion.
        Assumption: it is always the latest suggestion.

        :param osci_data: oscillogram data
        """
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
            suggestions = json.load(f)
        assert osci_data.comp_name in list(suggestions.values())[0]
        assert len(suggestions.keys()) == 1
        dtc = list(suggestions.keys())[0]
        print("DTC to set heatmap for:", dtc)

    def get_osci_set_id(self, components_to_be_recorded: Dict[str, str]) -> str:
        """
        Retrieves the oscillogram set ID in case of a number of parallel recorded oscillograms (else empty string).

        :param components_to_be_recorded: components to be recorded (in parallel)
        :return: oscillogram set ID
        """
        osci_set_id = ""
        if len(components_to_be_recorded.keys()) > 1:
            osci_set_id = self.instance_gen.extend_knowledge_graph_with_parallel_rec_osci_set()
        return osci_set_id

    def process_oscillogram_recordings(
            self, oscillograms: List[OscillogramData], suggestion_list: Dict[str, Tuple[str, bool]],
            anomalous_components: List[str], non_anomalous_components: List[str],
            components_to_be_recorded: Dict[str, str], classification_instances: Dict[str, str]
    ) -> None:
        """
        Iteratively processes the oscillograms, i.e., classifies each recording and overlays heatmaps.

        :param oscillograms: oscillograms to be classified
        :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
        :param anomalous_components: list to be filled with anomalous components, i.e., detected anomalies
        :param non_anomalous_components: list to be filled with regular components, i.e., no anomalies
        :param components_to_be_recorded: tuple of recorded components
        :param classification_instances: generated classification instances
        """
        osci_set_id = self.get_osci_set_id(components_to_be_recorded)
        for osci_data in oscillograms:  # iteratively process parallel recorded oscilloscope recordings
            osci_id = self.instance_gen.extend_knowledge_graph_with_oscillogram(osci_data.time_series, osci_set_id)
            print(colored("\n\nclassifying:" + osci_data.comp_name, "green", "on_grey", ["bold"]))
            voltages = osci_data.time_series

            model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(osci_data.comp_name)
            if model is None:
                util.no_trained_model_available(osci_data, suggestion_list)
                continue
            (model, model_meta_info) = model  # not only obtain the model here, but also meta info
            voltages = util.preprocess_time_series_based_on_model_meta_info(model_meta_info, voltages)
            try:
                util.validate_keras_model(model)
            except ValueError as e:
                util.invalid_model(osci_data, suggestion_list, e)
                continue

            net_input = util.construct_net_input(model, voltages)
            prediction = model.predict(np.array([net_input]))
            num_classes = len(prediction[0])
            # addresses both models with one output neuron and those with several
            anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5
            pred_value = prediction.max() if num_classes > 1 else prediction[0][0]

            if anomaly:
                self.log_anomaly(pred_value)
                anomalous_components.append(osci_data.comp_name)
            else:
                self.log_regular(pred_value)
                non_anomalous_components.append(osci_data.comp_name)

            heatmaps = self.gen_heatmaps(net_input, model, prediction)
            res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"
            self.log_corresponding_dtc(osci_data)
            print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])

            # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
            heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
                "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
            )
            heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(voltages), osci_data.comp_name + res_str)
            self.data_provider.provide_heatmaps(heatmap_img, osci_data.comp_name + res_str)
            classification_id = self.instance_gen.extend_knowledge_graph_with_oscillogram_classification(
                anomaly, components_to_be_recorded[osci_data.comp_name], osci_data.comp_name, pred_value,
                model_meta_info["model_id"], osci_id, heatmap_id
            )
            classification_instances[osci_data.comp_name] = classification_id

    def perform_manual_classifications(
            self, components_to_be_manually_verified: Dict[str, str], classification_instances: Dict[str, str],
            anomalous_components: List[str], non_anomalous_components: List[str]
    ) -> None:
        """
        Classifies the subset of components that are to be classified manually.

        :param components_to_be_manually_verified: components to be verified manually
        :param classification_instances: dictionary of classification instances {comp: classification_ID}
        :param anomalous_components: list of anomalous components (to be extended)
        :param non_anomalous_components: list of regular components (to be extended)
        """
        for comp in components_to_be_manually_verified.keys():
            print(colored("\n\nmanual inspection of component " + comp, "green", "on_grey", ["bold"]))
            anomaly = self.data_accessor.get_manual_judgement_for_component(comp)
            classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                anomaly, components_to_be_manually_verified[comp], comp
            )
            classification_instances[comp] = classification_id
            if anomaly:
                anomalous_components.append(comp)
            else:
                non_anomalous_components.append(comp)

    @staticmethod
    def gen_classified_components_dict(
            non_anomalous_components: List[str], anomalous_components: List[str]
    ) -> Dict[str, bool]:
        """
        Generates the dictionary of classified components.

        :param non_anomalous_components: list of regular components
        :param anomalous_components: list of anomalous components
        :return: classified components dict ({comp: anomaly})
        """
        classified_components = {}
        for comp in non_anomalous_components:
            classified_components[comp] = False
        for comp in anomalous_components:
            classified_components[comp] = True
        return classified_components

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'CLASSIFY_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_no_more_comp")
        """
        self.log_state_info()
        components_to_be_recorded, components_to_be_manually_verified = self.perform_synchronized_sensor_recordings(
            userdata.suggestion_list
        )
        oscillograms = self.data_accessor.get_oscillograms_by_components(list(components_to_be_recorded.keys()))
        anomalous_components = []
        non_anomalous_components = []
        classification_instances = {}
        self.process_oscillogram_recordings(
            oscillograms, userdata.suggestion_list, anomalous_components, non_anomalous_components,
            components_to_be_recorded, classification_instances
        )
        self.perform_manual_classifications(
            components_to_be_manually_verified, classification_instances, anomalous_components, non_anomalous_components
        )
        classified_components = self.gen_classified_components_dict(non_anomalous_components, anomalous_components)
        userdata.classified_components = list(classification_instances.values())
        self.log_classification_actions(
            classified_components, list(components_to_be_manually_verified.keys()), classification_instances
        )
        # there are three options:
        #   1. there's only one recording at a time and thus only one classification
        #   2. there are as many parallel recordings as there are suspect components for the DTC
        #   3. there are multiple parallel recordings, but not as many as there are suspect components for the DTC
        # TODO: are there remaining suspect components? (atm every component is suggested each case)
        remaining_suspect_components = False

        if len(anomalous_components) == 0 and not remaining_suspect_components:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE", "no_anomaly_no_more_comp"
            ))
            return "no_anomaly_no_more_comp"
        elif len(anomalous_components) == 0 and remaining_suspect_components:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "SUGGEST_SUSPECT_COMPONENTS", "no_anomaly"
            ))
            return "no_anomaly"
        elif len(anomalous_components) > 0:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "detected_anomalies"
            ))
            return "detected_anomalies"
