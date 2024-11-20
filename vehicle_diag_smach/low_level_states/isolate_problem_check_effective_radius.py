#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import io
import json
import os
from typing import Union, List, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import smach
import torch
from PIL import Image
from matplotlib.lines import Line2D
from obd_ontology import knowledge_graph_query_tool
from obd_ontology import ontology_instance_generator
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from tensorflow import keras
from termcolor import colored
from tsai.all import get_attribution_map
from tsai.models.XCM import XCM

from vehicle_diag_smach import util
from vehicle_diag_smach.config import SESSION_DIR, SUGGESTION_SESSION_FILE, OSCI_SESSION_FILES, \
    CLASSIFICATION_LOG_FILE, FAULT_PATH_TMP_FILE, SELECTED_OSCILLOGRAMS, FINAL_DEMO_TEST_SAMPLES
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor
from vehicle_diag_smach.interfaces.rule_based_model import RuleBasedModel
from vehicle_diag_smach.rule_based_models.Lambdasonde import Lambdasonde
from vehicle_diag_smach.rule_based_models.Saugrohrdrucksensor import Saugrohrdrucksensor


class IsolateProblemCheckEffectiveRadius(smach.State):
    """
    State in the low-level SMACH that represents situations in which one or more anomalies have been detected, and the
    task is to isolate the defective components based on their effective radius (structural knowledge).
    """

    def __init__(
            self, data_accessor: DataAccessor, model_accessor: ModelAccessor, data_provider: DataProvider, kg_url: str
    ) -> None:
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param model_accessor: implementation of the model accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['isolated_problem', 'isolated_problem_remaining_DTCs'],
                             input_keys=['classified_components'],
                             output_keys=['fault_paths'])
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url)
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)
        self.data_accessor = data_accessor
        self.model_accessor = model_accessor
        self.data_provider = data_provider

    @staticmethod
    def create_session_data_dir() -> None:
        """
        Creates the session data directory.
        """
        osci_iso_session_dir = SESSION_DIR + "/" + OSCI_SESSION_FILES + "/"
        if not os.path.exists(osci_iso_session_dir):
            os.makedirs(osci_iso_session_dir)

    def get_model_and_metadata(
            self, affecting_comp: str, voltage_dfs: List[pd.DataFrame], sub_comp: bool = False
    ) -> Tuple[Union[keras.models.Model, torch.nn.Module], dict]:
        """
        Retrieves the trained model and the corresponding metadata.

        :param affecting_comp: vehicle component to retrieve trained model for
        :param voltage_dfs: list of oscillogram channels
        :param sub_comp: whether the model is supposed to classify a subcomponent (single chan)
        :return: tuple of trained model and corresponding metadata
        """
        if sub_comp:  # rule-based single chan classification
            model = self.model_accessor.get_rule_based_univariate_ts_classification_model_by_component(affecting_comp)
        elif len(voltage_dfs) > 1:  # multivariate
            model = self.model_accessor.get_torch_multivariate_ts_classification_model_by_component(affecting_comp)
        else:  # univariate
            model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(affecting_comp)
        (model, model_meta_info) = model

        if isinstance(model, keras.models.Model):
            try:
                util.validate_keras_model(model)
            except ValueError as e:
                print("invalid model for the signal (component) to be classified:", affecting_comp)
                print("error:", e, "\nadding it to the list of components to be verified manually..")
                # TODO: actually handle the case
        elif isinstance(model, torch.nn.Module):
            # TODO: potentially add torch model validation
            pass
        return model, model_meta_info

    def provide_heatmaps(
            self, affecting_comp: str, res_str: str, heatmaps: Dict[str, np.ndarray], voltages: List[float]
    ) -> None:
        """
        Provides the generated heatmaps via the data provider.

        :param affecting_comp: component to classify oscillogram for
        :param res_str: result string (classification result + score)
        :param heatmaps: heatmaps to be provided
        :param voltages: classified voltage values (time series)
        """
        title = affecting_comp + "_" + res_str
        # TODO: fake time vals -- actually just data points
        time_vals = [i for i in range(len(voltages))]
        heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(voltages), title, time_vals)
        self.data_provider.provide_heatmaps(heatmap_img, title)

    def classify_with_keras_model(
            self, model: keras.models.Model, voltage_dfs: List[pd.DataFrame], dtc: str, affecting_comp: str
    ) -> Tuple[bool, float, str]:
        voltages = list(voltage_dfs[0].to_numpy().flatten())
        net_input = util.construct_net_input(model, voltages)
        prediction = model.predict(np.array([net_input]))
        num_classes = len(prediction[0])
        pred_value = prediction.max() if num_classes > 1 else prediction[0][0]
        # addresses both models with one output neuron and those with several
        anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5

        heatmaps = util.gen_heatmaps(net_input, model, prediction)
        print("DTC to set heatmap for:", dtc, "\nheatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
        # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
        heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
            "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
        )
        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(prediction[0][0]) + "]"
        self.provide_heatmaps(affecting_comp, res_str, heatmaps, voltages)
        return anomaly, pred_value, heatmap_id

    def classify_with_torch_model(
            self, model: torch.nn.Module, voltage_dfs: List[pd.DataFrame], comp_name: str
    ) -> Tuple[bool, float, str]:
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
        heatmap_id = ""
        xcm_model = XCM(c_in=len(voltage_dfs), c_out=2, seq_len=multivariate_sample.shape[2])
        xcm_model.load_state_dict(model.state_dict())
        assert type(xcm_model) == XCM
        # XCM's builtin way of displaying heatmaps
        # xcm_model.show_gradcam(tensor, TensorCategory(pred_value), figsize=(1920, 1080))

        att_maps = get_attribution_map(
            xcm_model,
            [xcm_model.conv2dblock, xcm_model.conv1dblock],
            tensor,
            detach=True,
            apply_relu=True
        )
        att_maps[0] = (att_maps[0] - att_maps[0].min()) / (att_maps[0].max() - att_maps[0].min())
        att_maps[1] = (att_maps[1] - att_maps[1].min()) / (att_maps[1].max() - att_maps[1].min())

        var_attr_heatmaps = {"var. attr. map " + str(i): att_maps[0].numpy()[i] for i in range(len(voltage_dfs))}
        # plot_multi_chan_heatmaps_as_overlay(
        #     var_attr_heatmaps,
        #     tensor[0].numpy(),
        #     'test_plot',
        #     list(range(len(tensor[0, 0]))),
        #     True
        # )
        time_attr_heatmaps = {"time attr. map " + str(i): att_maps[1].numpy()[i] for i in range(len(voltage_dfs))}
        # plot_multi_chan_heatmaps_as_overlay(
        #     time_attr_heatmaps,
        #     tensor[0].numpy(),
        #     'test_plot',
        #     list(range(len(tensor[0, 0]))),
        #     False
        # )

        for i in range(len(var_attr_heatmaps)):
            heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
                "XCM GradCAM", var_attr_heatmaps["var. attr. map " + str(i)].tolist()
            )
        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"

        # TODO: could use actual time values instead of list(range(len(tensor[0, 0]))
        var_attr_heatmap_img = cam.gen_multi_chan_heatmaps_as_overlay(
            var_attr_heatmaps, tensor[0].numpy(), comp_name + res_str, list(range(len(tensor[0, 0])))
        )
        # TODO: could use actual time values instead of list(range(len(tensor[0, 0]))
        time_attr_heatmap_img = cam.gen_multi_chan_heatmaps_as_overlay(
            time_attr_heatmaps, tensor[0].numpy(), comp_name + res_str, list(range(len(tensor[0, 0])))
        )
        self.data_provider.provide_heatmaps(var_attr_heatmap_img, comp_name + res_str + "_var_attr")
        self.data_provider.provide_heatmaps(time_attr_heatmap_img, comp_name + res_str + "_time_attr")
        # TODO: generally, we would want to store all heatmap IDs, i.e., for all channels
        return anomaly, pred_value, heatmap_id

    def classify_component(
            self, affecting_comp: str, dtc: str, classification_reason: str, sub_comp: bool = False
    ) -> Union[Tuple[bool, str], None]:
        """
        Classifies the oscillogram for the specified vehicle component.

        :param affecting_comp: component to classify oscillogram for
        :param dtc: DTC the original component suggestion was based on
        :param classification_reason: reason for the classification (ID of another classification)
        :param sub_comp: whether a subcomponent is classified (super component otherwise)
        :return: tuple of whether an anomaly has been detected and the corresponding classification ID
        """
        self.create_session_data_dir()
        # in this state, there is only one component to be classified, but there could be several

        if sub_comp:
            # in case of a subcomponent, we already recorded the multivariate data
            # --> the data is stored under the name of the super component
            super_comp = self.qt.query_super_component(affecting_comp)[0]
            # read from selected oscillograms in session files
            path = SESSION_DIR + "/" + SELECTED_OSCILLOGRAMS + "/"
            comp_recordings = [f for f in os.listdir(path) if super_comp in f]
            signal, _ = preprocess.gen_multivariate_signal_from_csv(path + comp_recordings[0])
            oscillograms = [OscillogramData(signal, super_comp)]
        else:
            oscillograms = self.data_accessor.get_oscillograms_by_components([affecting_comp])

        assert len(oscillograms) == 1
        voltage_dfs = oscillograms[0].time_series
        # TODO: here, we need to distinguish between multivariate and univariate
        osci_id = self.instance_gen.extend_knowledge_graph_with_oscillogram(voltage_dfs)

        model, model_meta_info = self.get_model_and_metadata(affecting_comp, voltage_dfs, sub_comp)

        if sub_comp:
            channels_for_super_comp = self.qt.query_sub_components_by_component(super_comp)
            idx = channels_for_super_comp.index(affecting_comp)
            voltage_dfs = [voltage_dfs[idx]]

        for df in range(len(voltage_dfs)):
            processed_chan = util.preprocess_time_series_based_on_model_meta_info(
                model_meta_info, voltage_dfs[df].to_numpy()
            ).flatten()
            voltage_dfs[df] = pd.DataFrame(processed_chan)

        if isinstance(model, keras.models.Model):
            print("KERAS MODEL")
            anomaly, pred_value, heatmap_id = self.classify_with_keras_model(model, voltage_dfs, dtc, affecting_comp)

        elif isinstance(model, torch.nn.Module):
            print("TORCH MODEL")
            anomaly, pred_value, heatmap_id = self.classify_with_torch_model(model, voltage_dfs, affecting_comp)

        elif isinstance(model, RuleBasedModel):
            if isinstance(model, Lambdasonde) or isinstance(model, Saugrohrdrucksensor):
                anomaly = model.predict(voltage_dfs[0].to_numpy(), affecting_comp)
            else:
                anomaly = model.predict(voltage_dfs[0].to_numpy())
            # no prediction values / heatmaps in case of the rule-based models
            pred_value = 1.0
            heatmap_id = ""
        else:
            print("unknown model:", type(model))
            return

        if anomaly:
            util.log_anomaly(pred_value)
        else:
            util.log_regular(pred_value)

        affecting_comp = super_comp if sub_comp else affecting_comp
        classification_id = self.instance_gen.extend_knowledge_graph_with_oscillogram_classification(
            anomaly, classification_reason, affecting_comp, pred_value, model_meta_info['model_id'],
            osci_id, heatmap_id
        )
        return anomaly, classification_id

    def construct_complete_graph(
            self, graph: Dict[str, List[str]], components_to_process: List[str]
    ) -> Dict[str, List[str]]:
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

            # add sub-components
            affecting_comp += self.qt.query_sub_components_by_component(comp)

            components_to_process += affecting_comp
            graph[comp] = affecting_comp
        return self.construct_complete_graph(graph, components_to_process)

    @staticmethod
    def create_legend_line(color: str, **kwargs) -> Line2D:
        """
        Creates the edge representations for the plot legend.

        :param color: color for legend line
        :return: generated line representation
        """
        return Line2D([0, 1], [0, 1], color=color, **kwargs)

    @staticmethod
    def get_anomalous_paths_without_sub_components(key: str, anomalous_paths: Dict[str, List[List[str]]]):
        anomalous_paths_without_sub = []
        if key in anomalous_paths.keys():
            for path in anomalous_paths[key]:
                anomalous_paths_without_sub.append([comp for comp in path if comp[0] != "(" and comp[-1] != ")"])
        return anomalous_paths_without_sub

    def compute_causal_links(
            self, to_relations: List[str], key: str, anomalous_paths: Dict[str, List[List[str]]],
            from_relations: List[str]
    ) -> List[int]:
        """
        Computes the causal links in the subgraph of cause-effect relationships.

        :param to_relations: 'to relations' of the considered subgraph
        :param key: considered component
        :param anomalous_paths: (branching) paths to the root cause
        :param from_relations: 'from relations' of the considered subgraph
        :return: causal links in the subgraph
        """
        anomalous_paths_without_sub = self.get_anomalous_paths_without_sub_components(key, anomalous_paths)
        causal_links = []
        for i in range(len(to_relations)):
            if key in anomalous_paths.keys():
                for j in range(len(anomalous_paths[key])):
                    for k in range(len(anomalous_paths[key][j]) - 1):
                        # default causal link check
                        if (anomalous_paths[key][j][k] == from_relations[i]
                                and anomalous_paths[key][j][k + 1] == to_relations[i]):
                            causal_links.append(i)
                            break
                        # super components with subcomponents in between
                        if (j < len(anomalous_paths_without_sub) and k < len(anomalous_paths_without_sub[j]) - 1
                                and (anomalous_paths_without_sub[j][k] == from_relations[i]
                                     and anomalous_paths_without_sub[j][k + 1] == to_relations[i])):
                            causal_links.append(i)
                            break
                        # subcomponent check (from sub to super)
                        if (anomalous_paths[key][j][k][1:-1] == from_relations[i]
                                and anomalous_paths[key][j][k + 1] == to_relations[i]):
                            causal_links.append(i)
                            break
                        # subcomponent check (from super to sub)
                        if (anomalous_paths[key][j][k] == from_relations[i]
                                and anomalous_paths[key][j][k + 1][1:-1] == to_relations[i]):
                            causal_links.append(i)
                            break
        return causal_links

    @staticmethod
    def set_edge_properties(
            causal_links: List[int], to_relations: List[str], from_relations: List[str],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> Tuple[List[str], List[int]]:
        """
        Sets the edge properties for the causal graph, i.e., sets edge colors and widths.

        :param causal_links: causal links in the subgraph
        :param to_relations: 'to relations' of the considered subgraph
        :param from_relations: 'from relations' of the considered subgraph
        :param explicitly_considered_links: links that have been verified explicitly
        :return: tuple of edge colors and widths
        """
        colors = ["#056517" if i not in causal_links else "#bf1029" for i in range(len(to_relations))]
        for i in range(len(from_relations)):
            # if the from-to relation is not part of the actually considered links, it should be black
            if from_relations[i] not in explicitly_considered_links.keys() or to_relations[i] not in \
                    explicitly_considered_links[from_relations[i]]:
                colors[i] = "black"
        # atm no diff between causal and non-causal
        widths = [30 if i not in causal_links else 30 for i in range(len(to_relations))]
        return colors, widths

    def gen_causal_graph_visualizations(
            self, anomalous_paths: Dict[str, List[List[str]]], complete_graphs: Dict[str, Dict[str, List[str]]],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> List[Image.Image]:
        """
        Visualizes the causal graphs along with the actual paths to the root cause.

        :param anomalous_paths: the paths to the root cause
        :param complete_graphs: the causal graphs
        :param explicitly_considered_links: links that have been verified explicitly
        :return: causal graph visualizations
        """
        visualizations = []
        for key in anomalous_paths.keys():
            print("isolation results, i.e., causal path:\n", key, ":", anomalous_paths[key])
        for key in complete_graphs.keys():
            print("visualizing graph for component:", key, "\n")
            plt.figure(figsize=(50, 35))
            plt.title("Causal Graph (Network of Effective Connections) for " + key, fontsize=50, fontweight='bold')
            from_relations = [k for k in complete_graphs[key].keys() for _ in range(len(complete_graphs[key][k]))]
            to_relations = [complete_graphs[key][k] for k in complete_graphs[key].keys()]
            to_relations = [item for lst in to_relations for item in lst]
            causal_links = self.compute_causal_links(to_relations, key, anomalous_paths, from_relations)
            # widths no longer used below (atm) -- changed arrow style
            edge_colors, widths = self.set_edge_properties(
                causal_links, to_relations, from_relations, explicitly_considered_links
            )
            df = pd.DataFrame({'from': from_relations, 'to': to_relations})
            g = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())

            # different graph layouts for the two demo scenarios
            if "multivariate" in FINAL_DEMO_TEST_SAMPLES:
                pos = nx.bipartite_layout(
                    G=g,
                    align='horizontal',
                    nodes=list(g.nodes)[:1] + list(g.nodes)[6:]  # those are the nodes for the first partition
                )
            else:
                pos = nx.spring_layout(g, scale=1, seed=67)

            labels = {n: n.replace(" ", "\n") for n in g.nodes}

            # assumption: each sub component is part of KG
            sub_components = [node for node in g.nodes if len(self.qt.query_sub_component_by_name(node)) == 1]

            # visually distinguish sub components from components
            node_colors = ["#a8b3b5" if node in sub_components else "#596466" for node in g.nodes]
            # set default outline to black
            node_outlines = ["black" for _ in g.nodes]

            if len(anomalous_paths) > 0:
                for i, node in enumerate(g.nodes):
                    for k in anomalous_paths.keys():
                        for path in anomalous_paths[k]:
                            if node in sub_components:
                                if "(" + node + ")" in path:
                                    # set anomalous sub component outline to red
                                    node_outlines[i] = "#bf1029"
                            else:  # super component
                                if node in path:  # set anomalous super components to red
                                    node_colors[i] = "#f5cdcb"
                                    node_outlines[i] = "#bf1029"

            nx.draw(
                g, pos=pos, with_labels=False, node_size=95000, font_size=25, alpha=0.75, arrows=True,
                edge_color=edge_colors, node_color=node_colors, edgecolors=node_outlines,
                linewidths=12, arrowstyle='simple', arrowsize=100
            )
            nx.draw_networkx_labels(g, pos, labels=labels, font_size=25, font_color='black')
            # initial preview does not require the same legend
            if len(anomalous_paths.keys()) > 0 and len(explicitly_considered_links.keys()) > 0:
                legend_lines = [self.create_legend_line(clr, lw=20) for clr in
                                ['#bf1029', '#056517', 'black', '#f5cdcb', '#596466', '#a8b3b5']]
                labels = ["fault path", "non-anomalous links", "disregarded", "anomalous components",
                          "regular components", "sub components"]
                plt.legend(legend_lines, labels, fontsize=40, loc='center right')
            else:  # initial graph legend
                legend_lines = [self.create_legend_line(clr, lw=20) for clr in ["#596466", "#a8b3b5"]]
                labels = ["components", "sub components"]
                plt.legend(legend_lines, labels, fontsize=40, loc='center right')

            buf = io.BytesIO()  # create bytes object and save matplotlib fig into it
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image = Image.open(buf)  # create PIL image object
            visualizations.append(image)
        return visualizations

    @staticmethod
    def log_classification_action(comp: str, anomaly: bool, use_oscilloscope: bool, classification_id: str) -> None:
        """
        Logs the classification actions to the session directory.

        :param comp: classified component
        :param anomaly: whether an anomaly was identified
        :param use_oscilloscope: whether an oscilloscope recording was used for the classification
        :param classification_id: ID of the corresponding classification instance
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            log_file.extend([{
                comp: anomaly,
                "State": "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS",
                "Classification Type": "manual inspection" if not use_oscilloscope else "osci classification",
                "Classification ID": classification_id
            }])
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

    def retrieve_already_checked_components(self, classified_components: List[str]) -> Dict[str, Tuple[bool, str]]:
        """
        Retrieves the already checked components together with the corresponding results:
            {comp: (prediction, classification_id)}
        A prediction of "true" stands for a detected anomaly.

        :param classified_components: list of classified components (IDs)
        :return: dictionary of already checked components: {comp: (prediction, classification_id)}
        """
        already_checked_components = {}
        for classification_id in classified_components:
            sus_comp_resp = self.qt.query_suspect_component_by_classification(classification_id)
            assert len(sus_comp_resp) == 1
            comp_id = sus_comp_resp[0].split("#")[1]
            comp_name = self.qt.query_suspect_component_name_by_id(comp_id)[0]
            # the prediction is retrieved as a string, not boolean, thus the check
            pred = self.qt.query_prediction_by_classification(classification_id)[0].lower() == "true"
            already_checked_components[comp_name] = (pred, classification_id)
        return already_checked_components

    @staticmethod
    def read_dtc_suggestion(anomalous_comp: str) -> str:
        """
        Reads the DTC the component suggestion was based on - assumption: it is always the latest suggestion.

        :param anomalous_comp: component to read the DTC the suggestion was based on for
        :return: read DTC
        """
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
            suggestions = json.load(f)
        assert anomalous_comp in list(suggestions.values())[0]
        assert len(suggestions.keys()) == 1
        return list(suggestions.keys())[0]

    def visualize_initial_graph(
            self, anomalous_paths: Dict[str, List[List[str]]], complete_graphs: Dict[str, Dict[str, List[str]]],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> None:
        """
        Visualizes the initial graph (without highlighted edges / pre isolation).

        :param anomalous_paths: the paths to the root cause
        :param complete_graphs: the causal graphs
        :param explicitly_considered_links: links that have been verified explicitly
        """
        visualizations = self.gen_causal_graph_visualizations(
            anomalous_paths, complete_graphs, explicitly_considered_links
        )
        self.data_provider.provide_causal_graph_visualizations(visualizations)

    def retrieve_sus_comp(self, class_id: str) -> str:
        """
        Retrieves the anomalous suspect component specified by the provided classification ID.

        :param class_id: classification ID to retrieve component for
        :return: suspect component for specified classification ID
        """
        sus_comp_resp = self.qt.query_suspect_component_by_classification(class_id)
        assert len(sus_comp_resp) == 1
        comp_id = sus_comp_resp[0].split("#")[1]
        return self.qt.query_suspect_component_name_by_id(comp_id)[0]

    def create_sub_component_paths_for_initial_comp(
            self, causal_paths: List[List[str]], dtc: str, classification_reason: str,
            explicitly_considered_links: Dict[str, List[str]]
    ):
        sub_anomalies = []
        entry_comp = causal_paths[0][0]
        for sub_comp in self.qt.query_sub_components_by_component(entry_comp):
            classification_res = self.classify_component(sub_comp, dtc, classification_reason, True)
            (anomaly, classification_id) = classification_res
            if anomaly:
                sub_anomalies.append(sub_comp)
            explicitly_considered_links[entry_comp].append(sub_comp)

        if len(sub_anomalies) > 0:
            starter_path = causal_paths[0].copy()
            causal_paths[0].append("(" + sub_anomalies[0] + ")")
            for i in range(1, len(sub_anomalies)):
                tmp = starter_path.copy()
                tmp.append("(" + sub_anomalies[i] + ")")
                causal_paths.append(tmp)

    def classify_sub_components_for_anomaly(
            self, checked_comp: str, dtc: str, classification_reason: str,
    ) -> Tuple[List[str], List[str]]:
        # classify subcomponents for anomaly (univariate classification)
        sub_anomalies = []
        sub_regulars = []
        for sub_comp in self.qt.query_sub_components_by_component(checked_comp):
            classification_res = self.classify_component(sub_comp, dtc, classification_reason, True)
            if classification_res is None:
                print("issue in classifying sub-component", sub_comp)
            else:
                (anomaly, classification_id) = classification_res
                print("classification res for", sub_comp, ":", anomaly)
                if anomaly:
                    sub_anomalies.append(sub_comp)
                else:
                    sub_regulars.append(sub_comp)
        return sub_anomalies, sub_regulars

    @staticmethod
    def create_sub_comp_paths_and_branch(prev_path: List[str], sub_anomalies: List[str], causal_paths: List[List[str]]):
        starter_path = prev_path.copy()
        if len(sub_anomalies) > 0:
            prev_path.append("(" + sub_anomalies[0] + ")")
            causal_paths.append(prev_path)
            # create sub-comp paths
            for i in range(1, len(sub_anomalies)):
                tmp = starter_path.copy()
                tmp.append("(" + sub_anomalies[i] + ")")
                causal_paths.append(tmp)

    @staticmethod
    def create_sub_comp_paths(causal_paths: List[List[str]], idx: int, sub_anomalies: List[str]):
        starter_path = causal_paths[idx].copy()
        if len(sub_anomalies) > 0:
            causal_paths[idx].append("(" + sub_anomalies[0] + ")")
            # create sub-comp paths
            for i in range(1, len(sub_anomalies)):
                tmp = starter_path.copy()
                tmp.append("(" + sub_anomalies[i] + ")")
                causal_paths.append(tmp)

    def handle_anomaly(
            self, causal_paths: List[List[str]], checked_comp: str, unisolated_anomalous_components: List[str],
            explicitly_considered_links: Dict[str, List[str]], dtc: str, classification_reason: str
    ) -> None:
        """
        Handles anomaly cases, i.e., extends the causal path, unisolated anomalous components, and explicitly
        considered links.

        :param causal_paths: causal paths to be extended
        :param checked_comp: checked component (found anomaly)
        :param unisolated_anomalous_components: list of unisolated anomalous components to be extended
        :param explicitly_considered_links: list of explicitly considered links to be extended
        :param dtc: DTC the original component suggestion was based on
        :param classification_reason: reason for the classification (ID of another classification)
        """
        sub_anomalies, sub_regulars = self.classify_sub_components_for_anomaly(checked_comp, dtc, classification_reason)

        already_in_path = False
        for i in range(len(causal_paths)):
            if checked_comp in causal_paths[i]:
                already_in_path = True

        if not already_in_path:
            found_link_in_path = False
            path_indices = []
            for i in range(len(causal_paths)):
                last_comp = causal_paths[i][-1]
                if "(" in last_comp:
                    last_comp = causal_paths[i][-2]
                if checked_comp in explicitly_considered_links[last_comp]:
                    found_link_in_path = True
                    path_indices.append(i)
            if found_link_in_path:  # extend path
                for idx in path_indices:
                    causal_paths[idx].append(checked_comp)
                    self.create_sub_comp_paths(causal_paths, idx, sub_anomalies)
            else:  # branch
                for i in range(len(causal_paths)):
                    second_last_comp = causal_paths[i][-2]
                    if checked_comp in explicitly_considered_links[second_last_comp]:
                        # this thing has to branch
                        prev_path = causal_paths[i][:len(causal_paths[i]) - 1].copy()
                        prev_path.append(checked_comp)
                        self.create_sub_comp_paths_and_branch(prev_path, sub_anomalies, causal_paths)

        affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(checked_comp)
        print("component potentially affected by:", affecting_comps)
        unisolated_anomalous_components += affecting_comps
        explicitly_considered_links[checked_comp] += affecting_comps.copy()
        explicitly_considered_links[checked_comp] += sub_regulars
        explicitly_considered_links[checked_comp] += sub_anomalies

    def work_through_unisolated_components(
            self, unisolated_anomalous_comps: List[str], explicitly_considered_links: Dict[str, List[str]],
            already_checked_comps: Dict[str, Tuple[bool, str]], causal_paths: List[List[str]], dtc: str,
            anomalous_comp: str
    ) -> None:
        """
        Works through the unisolated components, i.e., performs fault isolation.

        :param unisolated_anomalous_comps: unisolated anomalous components to work though
        :param explicitly_considered_links: list of explicitly considered links
        :param already_checked_comps: previously checked components (used to avoid redundant classifications)
        :param causal_paths: causal paths to be extended
        :param dtc: DTC the original component suggestion was based on
        :param anomalous_comp: initial anomalous component (entry point)
        """
        # the very first component entered; the beginning of isolation
        if len(causal_paths) == 1 and len(causal_paths[0]) == 1:
            self.create_sub_component_paths_for_initial_comp(
                causal_paths, dtc, already_checked_comps[anomalous_comp][1], explicitly_considered_links
            )
        while len(unisolated_anomalous_comps) > 0:
            comp_to_be_checked = unisolated_anomalous_comps.pop(0)
            print(colored("\ncomponent to be checked: " + comp_to_be_checked, "green", "on_grey", ["bold"]))
            if comp_to_be_checked not in list(explicitly_considered_links.keys()):
                explicitly_considered_links[comp_to_be_checked] = []
            if comp_to_be_checked in already_checked_comps.keys():
                print("already checked this component - anomaly:",
                      already_checked_comps[comp_to_be_checked][0])
                if already_checked_comps[comp_to_be_checked][0]:
                    self.handle_anomaly(
                        causal_paths, comp_to_be_checked, unisolated_anomalous_comps, explicitly_considered_links, dtc,
                        already_checked_comps[anomalous_comp][1]
                    )
                continue
            use_oscilloscope = self.qt.query_oscilloscope_usage_by_suspect_component(comp_to_be_checked)[0]
            if use_oscilloscope:
                print("use oscilloscope..")
                classification_res = self.classify_component(
                    comp_to_be_checked, dtc, already_checked_comps[anomalous_comp][1]
                )
                if classification_res is None:
                    anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                    classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                        anomaly, already_checked_comps[anomalous_comp][1], comp_to_be_checked
                    )
                else:
                    (anomaly, classification_id) = classification_res
                already_checked_comps[comp_to_be_checked] = (anomaly, classification_id)
            else:
                anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                    anomaly, already_checked_comps[anomalous_comp][1], comp_to_be_checked
                )
                already_checked_comps[comp_to_be_checked] = (anomaly, classification_id)
            if anomaly:
                self.handle_anomaly(
                    causal_paths, comp_to_be_checked, unisolated_anomalous_comps, explicitly_considered_links, dtc,
                    already_checked_comps[anomalous_comp][1]
                )
            self.log_classification_action(comp_to_be_checked, bool(anomaly), use_oscilloscope, classification_id)

    @staticmethod
    def create_tmp_file_for_already_found_fault_paths(fault_paths: Dict[str, List[List[str]]]) -> None:
        """
        Creates a temporary file for already found fault paths.

        :param fault_paths: already found fault paths to be saved in session dir
        """
        with open(SESSION_DIR + "/" + FAULT_PATH_TMP_FILE, "w") as f:
            json.dump(fault_paths, f, default=str)

    @staticmethod
    def load_already_found_fault_paths() -> Dict[str, List[List[str]]]:
        """
        Loads the already found fault paths from the tmp file.

        :return: already found fault paths
        """
        path = SESSION_DIR + "/" + FAULT_PATH_TMP_FILE
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS' state.
        Implements the search in the causal graph (cause-effect network).

        :param userdata: input of state
        :return: outcome of the state ("isolated_problem" | "isolated_problem_remaining_DTCs")
        """
        self.log_state_info()
        already_checked_components = self.retrieve_already_checked_components(userdata.classified_components)
        anomalous_paths = {}
        print(colored("constructing causal graph, i.e., subgraph of structural component knowledge..\n",
                      "green", "on_grey", ["bold"]))
        complete_graphs = {comp: self.construct_complete_graph({}, [comp])
                           for comp in already_checked_components.keys() if already_checked_components[comp][0]}
        explicitly_considered_links = {}
        self.visualize_initial_graph(anomalous_paths, complete_graphs, explicitly_considered_links)

        # important to compare to userdata here to not have a dictionary of changed size during iteration
        for class_id in userdata.classified_components:
            anomalous_comp = self.retrieve_sus_comp(class_id)
            if not already_checked_components[anomalous_comp][0]:
                continue
            print(colored("isolating " + anomalous_comp + "..", "green", "on_grey", ["bold"]))
            affecting_components = self.qt.query_affected_by_relations_by_suspect_component(anomalous_comp)

            if anomalous_comp not in list(explicitly_considered_links.keys()):
                explicitly_considered_links[anomalous_comp] = affecting_components.copy()
            else:
                explicitly_considered_links[anomalous_comp] += affecting_components.copy()

            print("component potentially affected by:", affecting_components)
            unisolated_anomalous_components = affecting_components
            causal_paths = [[anomalous_comp]]
            self.work_through_unisolated_components(
                unisolated_anomalous_components, explicitly_considered_links, already_checked_components, causal_paths,
                self.read_dtc_suggestion(anomalous_comp), anomalous_comp
            )
            anomalous_paths[anomalous_comp] = causal_paths
        visualizations = self.gen_causal_graph_visualizations(
            anomalous_paths, complete_graphs, explicitly_considered_links
        )
        self.data_provider.provide_causal_graph_visualizations(visualizations)
        remaining_dtc_instances = util.load_dtc_instances()
        print("REMAINING DTCs:", remaining_dtc_instances)
        if len(remaining_dtc_instances) > 0:
            self.create_tmp_file_for_already_found_fault_paths(anomalous_paths)  # write anomalous paths to session file
            self.data_provider.provide_state_transition(StateTransition(
                "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "SELECT_BEST_UNUSED_DTC_INSTANCE",
                "isolated_problem_remaining_DTCs"
            ))
            return "isolated_problem_remaining_DTCs"

        # load potential previous paths from session files
        already_found_fault_paths = self.load_already_found_fault_paths()
        already_found_fault_paths.update(anomalous_paths)  # merge dictionaries (already found + new ones)
        userdata.fault_paths = already_found_fault_paths
        self.data_provider.provide_state_transition(StateTransition(
            "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "PROVIDE_DIAG_AND_SHOW_TRACE", "isolated_problem"
        ))
        return "isolated_problem"
