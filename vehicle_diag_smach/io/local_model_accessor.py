#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import Union, Tuple

import torch
from obd_ontology import knowledge_graph_query_tool
from tensorflow import keras

from vehicle_diag_smach.config import TRAINED_MODEL_POOL, FINAL_DEMO_MODELS, KG_URL
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor
from vehicle_diag_smach.interfaces.rule_based_model import RuleBasedModel
from vehicle_diag_smach.rule_based_models.Lambdasonde import Lambdasonde
from vehicle_diag_smach.rule_based_models.Saugrohrdrucksensor import Saugrohrdrucksensor


class LocalModelAccessor(ModelAccessor):
    """
    Implementation of the model accessor interface using local model files.
    """

    def __init__(self):
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=KG_URL)

    def get_keras_univariate_ts_classification_model_by_component(
            self, component: str
    ) -> Union[Tuple[keras.models.Model, dict], None]:
        """
        Retrieves a trained model to classify signals of the specified vehicle component.

        The provided model is expected to be a Keras model satisfying the following assumptions:
            - input_shape: (None, len_of_ts, 1)
            - output_shape: (None, 1)
        Thus, in both cases we have a variable batch size due to `None`. For the input we expect a list of scalars and
        for the output exactly one scalar.

        :param component: vehicle component to retrieve trained model for
        :return: trained model and model meta info dictionary or `None` if unavailable
        """
        try:
            trained_model_file = TRAINED_MODEL_POOL + component + ".h5"
            print("loading trained model:", trained_model_file)
            # TODO: I could obtain these information from the KG
            #   - however, not with the deprecated prev demo KG
            model_meta_info = {
                "normalization_method": "z_norm",
                "model_id": "keras_univariate_ts_classification_model_001",
                "input_length": 23040
            }
            return keras.models.load_model(trained_model_file), model_meta_info
        except OSError as e:
            print("no trained model available for the signal (component) to be classified:", component)
            print("ERROR:", e)

    def get_rule_based_univariate_ts_classification_model_by_component(
            self, component: str
    ) -> Union[Tuple[RuleBasedModel, dict], None]:
        """
        Retrieves a rule-based model to classify signals of the specified vehicle component.

        The provided model is expected to be a rule-based model satisfying the following assumptions:
            - input_shape: (len_of_ts, 1)
            - output_shape: bool

        :param component: vehicle component to retrieve rule-based model for
        :return: rule-based model and model meta info dictionary or `None` if unavailable
        """
        super_component = self.qt.query_super_component(component)[0]
        # obtain meta info from the KG
        norm, model_id, input_len = self.qt.query_rule_based_model_meta_info_by_component(super_component)[0]
        model_meta_info = {"normalization_method": norm, "model_id": model_id, "input_length": int(input_len)}
        if "Lambdasonde" in component:
            return Lambdasonde(), model_meta_info
        elif "Saugrohrdrucksensor" in component:
            return Saugrohrdrucksensor(), model_meta_info
        return None

    def get_torch_multivariate_ts_classification_model_by_component(
            self, component: str
    ) -> Union[Tuple[torch.nn.Module, dict], None]:
        """
        Retrieves a trained model to classify signals of the specified vehicle component.

        The provided model is expected to be a Torch model satisfying the following assumptions:
            - input_shape: (None, len_of_ts, num_of_chan)
            - output_shape: (None, 1)
        Thus, in both cases we have a variable batch size due to `None`. For the input we expect a list of lists of
        scalars and for the output exactly one scalar.

        :param component: vehicle component to retrieve trained model for
        :return: trained model and model meta info dictionary or `None` if unavailable
        """
        try:
            trained_model_file = FINAL_DEMO_MODELS + component + ".pth"
            print("loading trained model:", trained_model_file)
            # obtain meta info from the KG
            norm, model_id, input_len = self.qt.query_xcm_model_meta_info_by_component(component)[0]
            model_meta_info = {"normalization_method": norm, "model_id": model_id, "input_length": int(input_len)}
            model = torch.load(trained_model_file)
            # ensure model is in evaluation mode
            model.eval()
            return model, model_meta_info
        except OSError as e:
            print("no trained model available for the signal (component) to be classified:", component)
            print("ERROR:", e)
