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
from PIL import Image
from matplotlib.lines import Line2D
from obd_ontology import knowledge_graph_query_tool
from obd_ontology import ontology_instance_generator
from oscillogram_classification import cam
from tensorflow import keras
from termcolor import colored

from vehicle_diag_smach import util
from vehicle_diag_smach.config import SESSION_DIR, SUGGESTION_SESSION_FILE, OSCI_SESSION_FILES, \
    CLASSIFICATION_LOG_FILE
from vehicle_diag_smach.data_types.state_transition import StateTransition
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


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
                             outcomes=['isolated_problem'],
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

    def get_model_and_metadata(self, affecting_comp: str) -> Tuple[keras.models.Model, dict]:
        """
        Retrieves the trained model and the corresponding metadata.

        :param affecting_comp: vehicle component to retrieve trained model for
        :return: tuple of trained model and corresponding metadata
        """
        model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(affecting_comp)
        if model is None:
            pass  # TODO: handle model is None cases
        (model, model_meta_info) = model
        try:
            util.validate_keras_model(model)
        except ValueError as e:
            print("invalid model for the signal (component) to be classified:", affecting_comp)
            print("error:", e, "\nadding it to the list of components to be verified manually..")
            # TODO: actually handle the case
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
        heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(voltages), title)
        self.data_provider.provide_heatmaps(heatmap_img, title)

    def classify_component(
            self, affecting_comp: str, dtc: str, classification_reason: str
    ) -> Union[Tuple[bool, str], None]:
        """
        Classifies the oscillogram for the specified vehicle component.

        :param affecting_comp: component to classify oscillogram for
        :param dtc: DTC the original component suggestion was based on
        :param classification_reason: reason for the classification (ID of another classification)
        :return: tuple of whether an anomaly has been detected and the corresponding classification ID
        """
        self.create_session_data_dir()
        # in this state, there is only one component to be classified, but there could be several
        oscillograms = self.data_accessor.get_oscillograms_by_components([affecting_comp])
        assert len(oscillograms) == 1
        voltages = oscillograms[0].time_series
        osci_id = self.instance_gen.extend_knowledge_graph_with_oscillogram(voltages)
        model, model_meta_info = self.get_model_and_metadata(affecting_comp)
        voltages = util.preprocess_time_series_based_on_model_meta_info(model_meta_info, voltages)
        net_input = util.construct_net_input(model, voltages)
        prediction = model.predict(np.array([net_input]))
        num_classes = len(prediction[0])
        # addresses both models with one output neuron and those with several
        anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5

        if anomaly:
            util.log_anomaly(prediction[0][0])
        else:
            util.log_regular(prediction[0][0])

        heatmaps = util.gen_heatmaps(net_input, model, prediction)
        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(prediction[0][0]) + "]"
        print("DTC to set heatmap for:", dtc)
        print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
        # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
        heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
            "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
        )
        self.provide_heatmaps(affecting_comp, res_str, heatmaps, voltages)
        classification_id = self.instance_gen.extend_knowledge_graph_with_oscillogram_classification(
            anomaly, classification_reason, affecting_comp, prediction[0][0], model_meta_info['model_id'],
            osci_id, heatmap_id
        )
        return np.argmax(prediction) == 0, classification_id

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
    def compute_causal_links(
            to_relations: List[str], key: str, anomalous_paths: Dict[str, List[str]], from_relations: List[str]
    ) -> List[int]:
        """
        Computes the causal links in the subgraph of cause-effect relationships.

        :param to_relations: 'to relations' of the considered subgraph
        :param key: considered component
        :param anomalous_paths: paths to the root cause
        :param from_relations: 'from relations' of the considered subgraph
        :return: causal links in the subgraph
        """
        causal_links = []
        for i in range(len(to_relations)):
            if key in anomalous_paths.keys():
                for j in range(len(anomalous_paths[key]) - 1):
                    # causal link check
                    if (anomalous_paths[key][j] == from_relations[i]
                            and anomalous_paths[key][j + 1] == to_relations[i]):
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
        colors = ['g' if i not in causal_links else 'r' for i in range(len(to_relations))]
        for i in range(len(from_relations)):
            # if the from-to relation is not part of the actually considered links, it should be black
            if from_relations[i] not in explicitly_considered_links.keys() or to_relations[i] not in \
                    explicitly_considered_links[from_relations[i]]:
                colors[i] = 'black'
        widths = [8 if i not in causal_links else 10 for i in range(len(to_relations))]
        return colors, widths

    def gen_causal_graph_visualizations(
            self, anomalous_paths: Dict[str, List[str]], complete_graphs: Dict[str, Dict[str, List[str]]],
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
            plt.figure(figsize=(25, 18))
            plt.title("Causal Graph (Network of Effective Connections) for " + key, fontsize=24, fontweight='bold')
            from_relations = [k for k in complete_graphs[key].keys() for _ in range(len(complete_graphs[key][k]))]
            to_relations = [complete_graphs[key][k] for k in complete_graphs[key].keys()]
            to_relations = [item for lst in to_relations for item in lst]
            causal_links = self.compute_causal_links(to_relations, key, anomalous_paths, from_relations)
            colors, widths = self.set_edge_properties(
                causal_links, to_relations, from_relations, explicitly_considered_links
            )
            df = pd.DataFrame({'from': from_relations, 'to': to_relations})
            g = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
            pos = nx.spring_layout(g, scale=0.3, seed=5)
            nx.draw(g, pos=pos, with_labels=True, node_size=30000, font_size=10, alpha=0.75, arrows=True,
                    edge_color=colors, width=widths)
            legend_lines = [self.create_legend_line(clr, lw=5) for clr in ['r', 'g', 'black']]
            labels = ["fault path", "non-anomalous links", "disregarded"]

            # initial preview does not require a legend
            if len(anomalous_paths.keys()) > 0 and len(explicitly_considered_links.keys()) > 0:
                plt.legend(legend_lines, labels, fontsize=20, loc='lower right')

            buf = io.BytesIO()  # create bytes object and save matplotlib fig into it
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image = Image.open(buf)  # create PIL image object
            visualizations.append(image)
        return visualizations

    @staticmethod
    def log_classification_action(comp: str, anomaly: bool, use_oscilloscope: bool, classification_id: str):
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
        print("############################################\n")

        # already checked components together with the corresponding results: {comp: (prediction, classification_id)}
        # a prediction of "true" -> anomaly
        already_checked_components = {}
        for classification_id in userdata.classified_components:
            sus_comp_resp = self.qt.query_suspect_component_by_classification(classification_id)
            assert len(sus_comp_resp) == 1
            comp_id = sus_comp_resp[0].split("#")[1]
            comp_name = self.qt.query_suspect_component_name_by_id(comp_id)[0]
            # the prediction is retrieved as a string, not boolean, thus the check
            pred = self.qt.query_prediction_by_classification(classification_id)[0] == "True"
            already_checked_components[comp_name] = (pred, classification_id)

        anomalous_paths = {}
        print(colored("constructing causal graph, i.e., subgraph of structural component knowledge..\n",
                      "green", "on_grey", ["bold"]))

        complete_graphs = {comp: self.construct_complete_graph({}, [comp])
                           for comp in already_checked_components.keys() if already_checked_components[comp][0]}
        explicitly_considered_links = {}

        # visualizing the initial graph (without highlighted edges / pre isolation)
        visualizations = self.gen_causal_graph_visualizations(
            anomalous_paths, complete_graphs, explicitly_considered_links
        )
        self.data_provider.provide_causal_graph_visualizations(visualizations)

        # important to compare to userdata here to not have a dictionary of changed size during iteration
        for class_id in userdata.classified_components:
            sus_comp_resp = self.qt.query_suspect_component_by_classification(class_id)
            assert len(sus_comp_resp) == 1
            comp_id = sus_comp_resp[0].split("#")[1]
            anomalous_comp = self.qt.query_suspect_component_name_by_id(comp_id)[0]
            if not already_checked_components[anomalous_comp][0]:
                continue

            # read DTC suggestion - assumption: it is always the latest suggestion
            with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
                suggestions = json.load(f)
            assert anomalous_comp in list(suggestions.values())[0]
            assert len(suggestions.keys()) == 1
            dtc = list(suggestions.keys())[0]

            print(colored("isolating " + anomalous_comp + "..", "green", "on_grey", ["bold"]))
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

                print(colored("\ncomponent to be checked: " + comp_to_be_checked, "green", "on_grey", ["bold"]))
                if comp_to_be_checked not in list(explicitly_considered_links.keys()):
                    explicitly_considered_links[comp_to_be_checked] = []

                if comp_to_be_checked in already_checked_components.keys():
                    print("already checked this component - anomaly:",
                          already_checked_components[comp_to_be_checked][0])

                    if already_checked_components[comp_to_be_checked][0]:
                        causal_path.append(comp_to_be_checked)
                        affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(comp_to_be_checked)
                        unisolated_anomalous_components += affecting_comps
                        explicitly_considered_links[comp_to_be_checked] += affecting_comps.copy()
                    continue

                use_oscilloscope = self.qt.query_oscilloscope_usage_by_suspect_component(comp_to_be_checked)[0]

                if use_oscilloscope:
                    print("use oscilloscope..")
                    classification_res = self.classify_component(
                        comp_to_be_checked, dtc, already_checked_components[anomalous_comp][1]
                    )
                    if classification_res is None:
                        anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                        classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                            anomaly, already_checked_components[anomalous_comp][1], comp_to_be_checked
                        )
                    else:
                        (anomaly, classification_id) = classification_res

                    already_checked_components[comp_to_be_checked] = (anomaly, classification_id)
                else:
                    anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                    classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                        anomaly, already_checked_components[anomalous_comp][1], comp_to_be_checked
                    )
                    already_checked_components[comp_to_be_checked] = (anomaly, classification_id)

                if anomaly:
                    causal_path.append(comp_to_be_checked)
                    affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(comp_to_be_checked)
                    print("component potentially affected by:", affecting_comps)
                    unisolated_anomalous_components += affecting_comps
                    explicitly_considered_links[comp_to_be_checked] += affecting_comps.copy()

                self.log_classification_action(comp_to_be_checked, bool(anomaly), use_oscilloscope, classification_id)

            anomalous_paths[anomalous_comp] = causal_path

        visualizations = self.gen_causal_graph_visualizations(
            anomalous_paths, complete_graphs, explicitly_considered_links
        )
        self.data_provider.provide_causal_graph_visualizations(visualizations)

        userdata.fault_paths = anomalous_paths
        self.data_provider.provide_state_transition(StateTransition(
            "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "PROVIDE_DIAG_AND_SHOW_TRACE", "isolated_problem"
        ))
        return "isolated_problem"
