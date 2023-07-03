#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os

import numpy as np
import smach
from obd_ontology import expert_knowledge_enhancer
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from termcolor import colored

from vehicle_diag_smach import util
from vehicle_diag_smach.config import SESSION_DIR, Z_NORMALIZATION, SUGGESTION_SESSION_FILE, CLASSIFICATION_LOG_FILE, \
    KG_URL
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class ClassifyComponents(smach.State):
    """
    State in the low-level SMACH that represents situations in which the suggested physical components in the vehicle
    are classified:
        - synchronized sensor recordings are performed at the suggested suspect components
        - recorded oscillograms are classified using the trained neural net model, i.e., detecting anomalies
        - manual inspection of suspect components, for which oscilloscope diagnosis is not appropriate, is performed
    """

    def __init__(self, model_accessor: ModelAccessor, data_accessor: DataAccessor, data_provider: DataProvider):
        """
        Initializes the state.

        :param model_accessor: implementation of the model accessor interface
        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        """
        smach.State.__init__(self,
                             outcomes=['detected_anomalies', 'no_anomaly', 'no_anomaly_no_more_comp'],
                             input_keys=['suggestion_list'],
                             output_keys=['classified_components'])
        self.model_accessor = model_accessor
        self.data_accessor = data_accessor
        self.data_provider = data_provider

    @staticmethod
    def log_classification_action(classified_components: dict, manually_inspected_components: list) -> None:
        """
        Logs the classification actions to the session directory.

        :param classified_components: dictionary of classified components + classification results
        :param manually_inspected_components: components that were classified manually by the mechanic
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            for k, v in classified_components.items():
                new_data = {
                    k: v,
                    "State": "CLASSIFY_COMPONENTS",
                    "Classification Type": "manual inspection"
                    if k in manually_inspected_components else "osci classification"
                }
                log_file.extend([new_data])
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'CLASSIFY_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_no_more_comp")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("CLASSIFY_COMPONENTS", "yellow", "on_grey", ["bold"]),
              "state (applying trained model)..")
        print("############################################")

        # perform synchronized sensor recordings
        components_to_be_recorded = [k for k, v in userdata.suggestion_list.items() if v]
        components_to_be_manually_verified = [k for k, v in userdata.suggestion_list.items() if not v]
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be verified manually:", components_to_be_manually_verified)
        print("------------------------------------------")

        print(colored("\nperform synchronized sensor recordings at:", "green", "on_grey", ["bold"]))
        for comp in components_to_be_recorded:
            print(colored("- " + comp, "green", "on_grey", ["bold"]))

        oscillograms = self.data_accessor.get_oscillograms_by_components(components_to_be_recorded)
        anomalous_components = []
        non_anomalous_components = []

        # iteratively process oscilloscope recordings
        for osci_data in oscillograms:
            print(colored("\n\nclassifying:" + osci_data.comp_name, "green", "on_grey", ["bold"]))
            voltages = osci_data.time_series

            # TODO: this depends on the trained model that is going to be applied
            #       -> we need to save this kind of meta information for each trained model we have on the platform
            #       -> all the preprocessing the model expects
            if Z_NORMALIZATION:
                voltages = preprocess.z_normalize_time_series(voltages)

            # TODO: we should probably not only obtain the model here, but also the meta info (see above)
            model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(osci_data.comp_name)
            if model is None:
                print("no trained model available for the signal (component) to be classified:", osci_data.comp_name)
                print("adding it to the list of components to be verified manually..")
                userdata.suggestion_list[osci_data.comp_name] = False
                continue
            (model, model_meta_info) = model
            print("model meta info:", model_meta_info)
            try:
                util.validate_keras_model(model)
            except ValueError as e:
                print("invalid model for the signal (component) to be classified:", osci_data.comp_name)
                print("error:", e)
                print("adding it to the list of components to be verified manually..")
                userdata.suggestion_list[osci_data.comp_name] = False
                continue

            net_input_size = model.layers[0].output_shape[0][1]
            assert net_input_size == len(voltages)
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
                anomalous_components.append(osci_data.comp_name)
            else:
                print("#####################################")
                print(colored("--> NO ANOMALIES DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
                print("#####################################")
                non_anomalous_components.append(osci_data.comp_name)

            heatmaps = {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
                        "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
                        "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
                        "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}

            res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"

            # read DTC suggestion - assumption: it is always the latest suggestion
            with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
                suggestions = json.load(f)
            assert osci_data.comp_name in list(suggestions.values())[0]
            assert len(suggestions.keys()) == 1
            dtc = list(suggestions.keys())[0]
            print("DTC to set heatmap for:", dtc)
            print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
            # extend KG with generated heatmap
            knowledge_enhancer = expert_knowledge_enhancer.ExpertKnowledgeEnhancer("", kg_url=KG_URL)
            # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
            knowledge_enhancer.extend_kg_with_heatmap_facts(
                heatmaps["tf-keras-gradcam"].tolist(), "tf-keras-gradcam"
            )
            heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, voltages, osci_data.comp_name + res_str)
            self.data_provider.provide_heatmaps(heatmap_img, osci_data.comp_name + res_str)

        # classifying the subset of components that are to be classified manually
        for comp in components_to_be_manually_verified:
            print(colored("\n\nmanual inspection of component " + comp, "green", "on_grey", ["bold"]))
            anomaly = self.data_accessor.get_manual_judgement_for_component(comp)
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
        self.log_classification_action(classified_components, components_to_be_manually_verified)

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
