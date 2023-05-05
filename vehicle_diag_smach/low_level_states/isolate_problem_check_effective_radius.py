#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import smach
from matplotlib.lines import Line2D
from obd_ontology import expert_knowledge_enhancer
from obd_ontology import knowledge_graph_query_tool
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from termcolor import colored

from vehicle_diag_smach.config import SESSION_DIR, OSCI_ISOLATION_SESSION_FILES, Z_NORMALIZATION, \
    SUGGESTION_SESSION_FILE
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor


class IsolateProblemCheckEffectiveRadius(smach.State):
    """
    State in the low-level SMACH that represents situations in which one or more anomalies have been detected, and the
    task is to isolate the defective components based on their effective radius (structural knowledge).
    """

    def __init__(self, data_accessor: DataAccessor, model_accessor: ModelAccessor):
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param model_accessor: implementation of the model accessor interface
        """
        smach.State.__init__(self,
                             outcomes=['isolated_problem'],
                             input_keys=['classified_components'],
                             output_keys=['fault_paths'])
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(local_kb=False)
        self.data_accessor = data_accessor
        self.model_accessor = model_accessor

    @staticmethod
    def manual_transition() -> None:
        val = None
        while val != "":
            val = input("\n..............................")

    def classify_component(self, affecting_comp: str, dtc: str) -> Union[bool, None]:
        """
        Classifies the oscillogram for the specified vehicle component.

        :param affecting_comp: component to classify oscillogram for
        :param dtc: DTC the original component suggestion was based on
        :return: whether an anomaly has been detected
        """
        # create session data directory
        osci_iso_session_dir = SESSION_DIR + "/" + OSCI_ISOLATION_SESSION_FILES + "/"
        if not os.path.exists(osci_iso_session_dir):
            os.makedirs(osci_iso_session_dir)

        val = None
        while val != "":
            val = input("\npress 'ENTER' when the recording phase is finished and the oscillogram is generated..")

        # TODO: in this state, there is only one component to be classified, but there could be several
        oscillograms = self.data_accessor.get_oscillograms_by_components([affecting_comp])
        assert len(oscillograms) == 1
        voltages = oscillograms[0].time_series

        if Z_NORMALIZATION:
            voltages = preprocess.z_normalize_time_series(voltages)

        model = self.model_accessor.get_model_by_component(affecting_comp)

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

        if anomaly:
            print("#####################################")
            print(colored("--> ANOMALY DETECTED (" + str(prediction[0][0]) + ")", "green", "on_grey", ["bold"]))
            print("#####################################\n")
        else:
            print("#####################################")
            print(colored("--> NO ANOMALIES DETECTED (" + str(prediction[0][0]) + ")", "green", "on_grey", ["bold"]))
            print("#####################################\n")

        heatmaps = {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
                    "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
                    "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
                    "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}

        res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(prediction[0][0]) + "]"

        print("DTC to set heatmap for:", dtc)
        print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])
        # extend KG with generated heatmap
        knowledge_enhancer = expert_knowledge_enhancer.ExpertKnowledgeEnhancer("")
        # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
        knowledge_enhancer.extend_kg_with_heatmap_facts(dtc, affecting_comp, heatmaps["tf-keras-gradcam"].tolist())
        title = affecting_comp + "_" + res_str
        cam.plot_heatmaps_as_overlay(heatmaps, voltages, title)

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
        for key in anomalous_paths.keys():
            print("isolation results, i.e., causal path:")
            print(key, ":", anomalous_paths[key])

        for key in complete_graphs.keys():
            print("visualizing graph for component:", key, "\n")
            plt.title("Causal Graph (Network of Effective Connections) for " + key, fontsize=24, fontweight='bold')

            from_relations = [k for k in complete_graphs[key].keys() for _ in range(len(complete_graphs[key][k]))]
            to_relations = [complete_graphs[key][k] for k in complete_graphs[key].keys()]
            to_relations = [item for lst in to_relations for item in lst]

            causal_links = []
            for i in range(len(to_relations)):
                if key in anomalous_paths.keys():
                    for j in range(len(anomalous_paths[key]) - 1):
                        # causal link check
                        if anomalous_paths[key][j] == from_relations[i] \
                                and anomalous_paths[key][j + 1] == to_relations[i]:
                            causal_links.append(i)
                            break

            colors = ['g' if i not in causal_links else 'r' for i in range(len(to_relations))]
            for i in range(len(from_relations)):
                # if the from-to relation is not part of the actually considered links, it should be black
                if from_relations[i] not in explicitly_considered_links.keys() or to_relations[i] not in \
                        explicitly_considered_links[from_relations[i]]:
                    colors[i] = 'black'

            widths = [5 if i not in causal_links else 10 for i in range(len(to_relations))]
            df = pd.DataFrame({'from': from_relations, 'to': to_relations})

            g = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
            pos = nx.spring_layout(g, seed=5)
            nx.draw(g, pos=pos, with_labels=True, node_size=40000, alpha=0.75, arrows=True, edge_color=colors,
                    width=widths)

            legend_lines = [self.create_legend_lines(clr, lw=5) for clr in ['r', 'g', 'black']]
            labels = ["fault path", "non-anomalous links", "disregarded"]
            # initial preview does not require a legend
            if len(anomalous_paths.keys()) > 0 and len(explicitly_considered_links.keys()) > 0:
                plt.legend(legend_lines, labels, fontsize=18)
            plt.show()

    @staticmethod
    def manual_inspection_of_component() -> bool:
        """
        Guides a manual inspection of the component by the mechanic.

        :return: true -> anomaly / false -> no anomaly
        """
        print("manual inspection of component (no oscilloscope)..")
        val = ""
        while val not in ['0', '1']:
            val = input("\npress '0' for defective component, i.e., anomaly, and '1' for no defect..")
        return val == "0"

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

        # already checked components together with the corresponding results (true -> anomaly)
        already_checked_components = userdata.classified_components.copy()
        anomalous_paths = {}
        print(colored("constructing causal graph, i.e., subgraph of structural component knowledge..\n",
                      "green", "on_grey", ["bold"]))

        complete_graphs = {comp: self.construct_complete_graph({}, [comp])
                           for comp in userdata.classified_components.keys() if userdata.classified_components[comp]}
        explicitly_considered_links = {}

        self.manual_transition()
        # visualizing the initial graph (without highlighted edges / pre isolation)
        self.visualize_causal_graphs(anomalous_paths, complete_graphs, explicitly_considered_links)

        for anomalous_comp in userdata.classified_components.keys():
            if not userdata.classified_components[anomalous_comp]:
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
                    anomaly = self.classify_component(comp_to_be_checked, dtc)
                    if anomaly is None:
                        anomaly = self.manual_inspection_of_component()
                    else:
                        already_checked_components[comp_to_be_checked] = anomaly
                else:
                    anomaly = self.manual_inspection_of_component()
                    already_checked_components[comp_to_be_checked] = anomaly

                if anomaly:
                    causal_path.append(comp_to_be_checked)
                    affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(comp_to_be_checked)
                    print("component potentially affected by:", affecting_comps)
                    unisolated_anomalous_components += affecting_comps
                    explicitly_considered_links[comp_to_be_checked] += affecting_comps.copy()

            anomalous_paths[anomalous_comp] = causal_path

        self.visualize_causal_graphs(anomalous_paths, complete_graphs, explicitly_considered_links)
        userdata.fault_paths = anomalous_paths
        return "isolated_problem"
