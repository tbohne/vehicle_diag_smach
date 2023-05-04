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
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OSCI_SESSION_FILES, Z_NORMALIZATION, SUGGESTION_SESSION_FILE
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class ClassifyOscillograms(smach.State):
    """
    State in the low-level SMACH that represents situations in which the recorded oscillograms are classified using
    the trained neural net model, i.e., detecting anomalies.
    """

    def __init__(self, model_accessor: ModelAccessor):
        """
        Initializes the state.

        :param model_accessor: implementation of the model accessor interface
        """

        smach.State.__init__(self,
                             outcomes=['detected_anomalies', 'no_anomaly',
                                       'no_anomaly_no_more_comp'],
                             input_keys=['suggestion_list'],
                             output_keys=['classified_components'])

        self.model_accessor = model_accessor

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'CLASSIFY_OSCILLOGRAMS' state.

        :param userdata: input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_no_more_comp")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("CLASSIFY_OSCILLOGRAMS", "yellow", "on_grey", ["bold"]),
              "state (applying trained model)..")
        print("############################################")

        anomalous_components = []
        non_anomalous_components = []

        # iteratively process oscilloscope recordings
        for osci_path in Path(SESSION_DIR + "/" + OSCI_SESSION_FILES + "/").rglob('*.csv'):
            label = str(osci_path).split("/")[2].replace(".csv", "")
            comp_name = label.split("_")[-1]

            print(colored("\n\nclassifying:" + comp_name, "green", "on_grey", ["bold"]))
            _, voltages = preprocess.read_oscilloscope_recording(osci_path)

            if Z_NORMALIZATION:
                voltages = preprocess.z_normalize_time_series(voltages)

            model = self.model_accessor.get_model_by_component(comp_name)
            if model is None:
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
            return "no_anomaly_no_more_comp"
        elif len(anomalous_components) == 0 and remaining_suspect_components:
            return "no_anomaly"
        elif len(anomalous_components) > 0:
            return "detected_anomalies"
