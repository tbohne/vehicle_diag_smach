#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import smach
import torch
from obd_ontology import ontology_instance_generator, knowledge_graph_query_tool
from oscillogram_classification import cam
from tensorflow import keras
from termcolor import colored
from tsai.all import get_attribution_map
from tsai.models.XCM import XCM

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

    def __init__(
            self, model_accessor: ModelAccessor, data_accessor: DataAccessor, data_provider: DataProvider, kg_url: str
    ) -> None:
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
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)

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

    def perform_synchronized_sensor_recordings(
            self, suggestion_list: Dict[str, Tuple[str, bool]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Performs synchronized sensor recordings based on the provided suggestion list.

        :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
        :return: tuple of components to be recorded and components to be verified manually
                 ({comp: reason_for}, {comp: reason_for})
        """
        components_to_be_recorded = {k: v[0] for k, v in suggestion_list.items() if v[1]}
        if len(components_to_be_recorded) > 0:
            oscillograms = self.data_accessor.get_oscillograms_by_components(list(components_to_be_recorded.keys()))
            multivariate = True if len(oscillograms[0].time_series) > 1 else False
        components_to_be_manually_verified = {k: v[0] for k, v in suggestion_list.items() if not v[1]}
        channels = {}
        for component in components_to_be_recorded.keys():
            if multivariate:
                norm, model_id, input_len = self.qt.query_xcm_model_meta_info_by_component(component)[0]
                model_instance = self.qt.query_model_by_model_id(model_id)[0]
                model_uuid = model_instance.split("#")[1]
                input_chan_req_resp = self.qt.query_input_chan_req_by_model(model_uuid)
                assert len(input_chan_req_resp) > 0
                comp_channels = np.empty(len(input_chan_req_resp), dtype=object)
                for input_chan_req, req_idx in input_chan_req_resp:
                    input_chan_req_id = input_chan_req.split("#")[1]
                    req_chan = self.qt.query_channel_by_input_req(input_chan_req_id)
                    assert len(req_chan) == 1
                    req_chan_name = req_chan[0][1]
                    comp_channels[int(req_idx)] = req_chan_name
                channels[component] = comp_channels
            else:
                channels[component] = [component + "-Signal"]
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be verified manually:", components_to_be_manually_verified)
        print("------------------------------------------")
        print(colored("\nperform synchronized sensor recordings at:", "green", "on_grey", ["bold"]))
        for comp, channel_names in channels.items():
            print(colored(f"- {comp}: channels to be recorded - {list(channel_names)} ", "green", "on_grey", ["bold"]))
        return components_to_be_recorded, components_to_be_manually_verified

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

    def classify_with_keras_model(
            self, model: keras.models.Model, voltage_dfs: List[pd.DataFrame], comp_name: str
    ) -> Tuple[bool, float, str]:
        """
        Classifies the provided voltage dataframes using the provided Keras model.

        :param model: trained Keras model to classify voltage frames
        :param voltage_dfs: voltage data to be classified
        :param comp_name: name of the corresponding component
        :return: (anomaly, prediction value, heatmap ID)
        """
        voltages = list(voltage_dfs[0].to_numpy().flatten())
        net_input = util.construct_net_input(model, voltages)
        # TODO: fake time vals -- actually just data points
        time_vals = [i for i in range(len(voltages))]
        prediction = model.predict(np.array([net_input]))

        num_classes = len(prediction[0])
        # addresses both models with one output neuron and those with several
        anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5
        pred_value = prediction.max() if num_classes > 1 else prediction[0][0]

        heatmaps = util.gen_heatmaps(net_input, model, prediction)
        print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
        # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
        heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
            "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
        )
        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"
        heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(voltages), comp_name + res_str, time_vals)
        self.data_provider.provide_heatmaps(heatmap_img, comp_name + res_str)
        return anomaly, pred_value, heatmap_id

    def classify_with_torch_model(
            self, model: torch.nn.Module, voltage_dfs: List[pd.DataFrame], comp_name: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Classifies the provided voltage dataframes using the provided torch model.

        :param model: trained torch model to classify voltage frames
        :param voltage_dfs: voltage data to be classified
        :param comp_name: name of the corresponding component
        :return: (anomaly, prediction value, list of heatmap IDs)
        """
        multivariate_sample = np.array([df.to_numpy() for df in voltage_dfs])
        # expected shape for test signals: (1, chan, length)
        multivariate_sample = multivariate_sample.reshape(
            multivariate_sample.shape[2], multivariate_sample.shape[0], multivariate_sample.shape[1]
        )
        tensor = torch.from_numpy(multivariate_sample).float()
        # assumes model outputs logits for a multi-class classification problem
        logits = model(tensor)
        # convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=1)
        num_classes = len(probas[0])

        # addresses both models with one output neuron and those with several
        anomaly = int(torch.argmax(probas, dim=1)) == 0 if num_classes > 1 else probas[0][0] <= 0.5
        pred_value = float(probas.max()) if num_classes > 1 else probas[0][0]

        # heatmap generation for torch model (XCM)
        heatmap_ids = []
        xcm_model = XCM(c_in=len(voltage_dfs), c_out=2, seq_len=multivariate_sample.shape[2])
        xcm_model.load_state_dict(model.state_dict())
        assert type(xcm_model) == XCM
        # XCM's builtin way of displaying heatmaps
        # xcm_model.show_gradcam(tensor, TensorCategory(pred_value), figsize=(1920, 1080))

        att_maps = get_attribution_map(
            xcm_model, [xcm_model.conv2dblock, xcm_model.conv1dblock], tensor, detach=True, apply_relu=True
        )
        att_maps[0] = (att_maps[0] - att_maps[0].min()) / (att_maps[0].max() - att_maps[0].min())
        att_maps[1] = (att_maps[1] - att_maps[1].min()) / (att_maps[1].max() - att_maps[1].min())

        var_attr_heatmaps = {"var. attr. map " + str(i): att_maps[0].numpy()[i] for i in range(len(voltage_dfs))}
        # plot_multi_chan_heatmaps_as_overlay(
        #     var_attr_heatmaps, tensor[0].numpy(), 'test_plot', list(range(len(tensor[0, 0]))), True
        # )
        time_attr_heatmaps = {"time attr. map " + str(i): att_maps[1].numpy()[i] for i in range(len(voltage_dfs))}
        # plot_multi_chan_heatmaps_as_overlay(
        #     time_attr_heatmaps, tensor[0].numpy(), 'test_plot', list(range(len(tensor[0, 0]))), False
        # )

        for i in range(len(var_attr_heatmaps)):
            heatmap_ids.append(
                self.instance_gen.extend_knowledge_graph_with_heatmap(
                    "XCM GradCAM variable attribution map", var_attr_heatmaps["var. attr. map " + str(i)].tolist()
                )
            )
        # add only one of the time attribution maps, as they are identical for all channels
        heatmap_ids.append(
            self.instance_gen.extend_knowledge_graph_with_heatmap(
                "XCM GradCAM time attribution map", time_attr_heatmaps["time attr. map " + str(0)].tolist()
            )
        )
        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"

        var_attr_heatmap_img = cam.gen_multi_chan_heatmaps_as_overlay(
            var_attr_heatmaps, tensor[0].numpy(), comp_name + res_str, list(range(len(tensor[0, 0])))
        )
        time_attr_heatmap_img = cam.gen_multi_chan_heatmaps_as_overlay(
            time_attr_heatmaps, tensor[0].numpy(), comp_name + res_str, list(range(len(tensor[0, 0])))
        )
        self.data_provider.provide_heatmaps(var_attr_heatmap_img, comp_name + res_str + "_var_attr")
        self.data_provider.provide_heatmaps(time_attr_heatmap_img, comp_name + res_str + "_time_attr")

        # store all heatmap IDs, i.e. for all channels
        return anomaly, pred_value, heatmap_ids

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
            voltage_dfs = osci_data.time_series
            assert isinstance(voltage_dfs[0], (pd.Series, pd.DataFrame, list))
            multivariate = True if len(voltage_dfs) > 1 else False
            print(colored("\n\nclassifying:" + osci_data.comp_name, "green", "on_grey", ["bold"]))

            if multivariate:
                model = self.model_accessor.get_torch_multivariate_ts_classification_model_by_component(
                    osci_data.comp_name
                )
            else:  # univariate
                model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(
                    osci_data.comp_name
                )

            if model is None:
                util.no_trained_model_available(osci_data, suggestion_list)
                continue
            (model, model_meta_info) = model  # not only obtain the model here, but also meta info

            osci_ids = []
            for df in range(len(voltage_dfs)):
                osci_ids.append(
                    self.instance_gen.extend_knowledge_graph_with_oscillogram(voltage_dfs[df], osci_set_id)
                )
                processed_chan = util.preprocess_time_series_based_on_model_meta_info(
                    model_meta_info, voltage_dfs[df].to_numpy()
                ).flatten()
                voltage_dfs[df] = pd.DataFrame(processed_chan)

            if isinstance(model, torch.nn.Module):
                print("TORCH MODEL")
                # TODO: potentially add torch model validation
                anomaly, pred_value, heatmap_ids = self.classify_with_torch_model(
                    model, voltage_dfs, osci_data.comp_name
                )
            elif isinstance(model, keras.models.Model):
                print("KERAS MODEL")
                try:
                    util.validate_keras_model(model)
                except ValueError as e:
                    util.invalid_model(osci_data, suggestion_list, e)
                    continue
                anomaly, pred_value, heatmap_id = self.classify_with_keras_model(
                    model, voltage_dfs, osci_data.comp_name
                )
                heatmap_ids = [heatmap_id]
            else:
                print("unknown model:", type(model))
                continue

            if anomaly:
                util.log_anomaly(pred_value)
                anomalous_components.append(osci_data.comp_name)
            else:
                util.log_regular(pred_value)
                non_anomalous_components.append(osci_data.comp_name)

            self.log_corresponding_dtc(osci_data)
            classification_id = self.instance_gen.extend_knowledge_graph_with_oscillogram_classification(
                anomaly, components_to_be_recorded[osci_data.comp_name], osci_data.comp_name, pred_value,
                model_meta_info["model_id"], osci_ids, heatmap_ids
            )
            classification_instances[osci_data.comp_name] = classification_id

            # "overlays" relation between heatmap and oscillogram
            if len(heatmap_ids) == len(osci_ids):
                for ind in range(len(osci_ids)):
                    self.instance_gen.extend_knowledge_graph_with_overlays_relation(
                        heatmap_id=heatmap_ids[ind], osci_id=osci_ids[ind]
                    )
            elif len(heatmap_ids) == len(osci_ids) + 1:
                # it is expected that the last heatmap is a time attribution map that overlays all heatmaps
                for ind in range(len(osci_ids)):
                    self.instance_gen.extend_knowledge_graph_with_overlays_relation(
                        heatmap_id=heatmap_ids[ind], osci_id=osci_ids[ind]
                    )
                    self.instance_gen.extend_knowledge_graph_with_overlays_relation(
                        heatmap_id=heatmap_ids[-1], osci_id=osci_ids[ind]
                    )
            else:
                print("Number of heatmaps does not match number of oscillograms.")

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
