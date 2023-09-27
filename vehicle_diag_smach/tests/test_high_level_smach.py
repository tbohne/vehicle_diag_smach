#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import logging
import os
import unittest

import smach
import tensorflow as tf
from obd_ontology import knowledge_graph_query_tool

from vehicle_diag_smach.config import KG_URL
from vehicle_diag_smach.high_level_smach import VehicleDiagnosisStateMachine
from vehicle_diag_smach.io.local_model_accessor import LocalModelAccessor
from vehicle_diag_smach.io.test_data_accessor import TestDataAccessor
from vehicle_diag_smach.io.test_data_provider import TestDataProvider
from vehicle_diag_smach.util import log_info, log_debug, log_warn, log_err


class TestHighLevelStateMachine(unittest.TestCase):
    """
    Multiple test scenarios to ensure basic functionality of the entire diagnostic process.
    """

    def test_model_availability_for_scenario_zero(self) -> None:
        """
        Tests the availability of expected trained models for test scenario zero.
        """
        for i in range(1, 14, 2):  # all odd components are expected to have a trained model
            self.assertTrue(os.path.exists("res/trained_model_pool/C" + str(i) + ".h5"))

    def test_hosted_knowledge_graph_for_scenario_zero(self) -> None:
        """
        Tests whether the correct (expected) KG is hosted for the unit tests:
        https://github.com/tbohne/obd_ontology/tree/main/knowledge_base/unit_test_kg.nt.
        Essentially, it verifies that the KG contains the assumed knowledge for test scenario zero.
        """
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=KG_URL)

        for i in range(1, 14):  # all expected components have to exist in the KG
            self.assertEqual(len(qt.query_suspect_component_by_name("C" + str(i))), 1)

        # check expected `affected_by` relations
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C1")),
                             sorted(["C7", "C2", "C6"]))
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C2")), sorted(["C8", "C3"]))
        self.assertListEqual(qt.query_affected_by_relations_by_suspect_component("C8"), ["C12"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C3")), sorted(["C4", "C9"]))
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C4")),
                             sorted(["C11", "C10", "C5"]))
        self.assertListEqual(qt.query_affected_by_relations_by_suspect_component("C11"), ["C13"])

        # expected entry component
        self.assertListEqual(qt.query_suspect_components_by_dtc("P0125"), ["C1"])

    def test_complete_diagnosis_scenario_zero(self) -> None:
        """
        Tests the state machine's functionality based on the defined test "scenario zero" with its expected outcome.
        """
        # init local implementations of I/O interfaces
        data_acc = TestDataAccessor(0)  # scenario zero
        data_prov = TestDataProvider()
        model_acc = LocalModelAccessor()

        sm = VehicleDiagnosisStateMachine(data_acc, model_acc, data_prov)
        sm.execute()
        self.assertEqual(sm.userdata.final_output, ['C5 -> C4 -> C3 -> C2 -> C1'])

    def test_model_availability_for_scenario_one(self) -> None:
        """
        Tests the availability of expected trained models for test scenario one.
        """
        for i in range(15, 22, 2):  # all odd components are expected to have a trained model
            self.assertTrue(os.path.exists("res/trained_model_pool/C" + str(i) + ".h5"))

    def test_hosted_knowledge_graph_for_scenario_one(self) -> None:
        """
        Tests whether the correct (expected) KG is hosted for the unit tests:
        https://github.com/tbohne/obd_ontology/tree/main/knowledge_base/unit_test_kg.nt.
        Essentially, it verifies that the KG contains the assumed knowledge for test scenario one.
        """
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=KG_URL)

        for i in range(14, 22):  # all expected components have to exist in the KG
            self.assertEqual(len(qt.query_suspect_component_by_name("C" + str(i))), 1)

        # check expected `affected_by` relations
        self.assertListEqual(qt.query_affected_by_relations_by_suspect_component("C14"), ["C15"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C15")),
                             sorted(["C16", "C17", "C18"]))
        self.assertListEqual(qt.query_affected_by_relations_by_suspect_component("C18"), ["C19"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C19")), sorted(["C20", "C21"]))

        # expected entry component
        self.assertListEqual(qt.query_suspect_components_by_dtc("P0126"), ["C14"])

    def test_complete_diagnosis_scenario_one(self) -> None:
        """
        Tests the state machine's functionality based on the defined test "scenario one" with its expected outcome.
        """
        # init local implementations of I/O interfaces
        data_acc = TestDataAccessor(1)  # scenario one
        data_prov = TestDataProvider()
        model_acc = LocalModelAccessor()

        sm = VehicleDiagnosisStateMachine(data_acc, model_acc, data_prov)
        sm.execute()
        self.assertEqual(
            sorted(sm.userdata.final_output),
            sorted(['C20 -> C19 -> C18 -> C15 -> C14', 'C21 -> C19 -> C18 -> C15 -> C14'])
        )

    def test_model_availability_for_scenario_two(self) -> None:
        """
        Tests the availability of expected trained models for test scenario zero.
        """
        for i in range(23, 36, 2):  # all odd components are expected to have a trained model
            self.assertTrue(os.path.exists("res/trained_model_pool/C" + str(i) + ".h5"))

    def test_hosted_knowledge_graph_for_scenario_two(self) -> None:
        """
        Tests whether the correct (expected) KG is hosted for the unit tests:
        https://github.com/tbohne/obd_ontology/tree/main/knowledge_base/unit_test_kg.nt.
        Essentially, it verifies that the KG contains the assumed knowledge for test scenario two.
        """
        qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=KG_URL)

        for i in range(22, 37):  # all expected components have to exist in the KG
            self.assertEqual(len(qt.query_suspect_component_by_name("C" + str(i))), 1)

        # check expected `affected_by` relations
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C22")), sorted(["C23", "C24"]))
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C24")), sorted(["C25", "C26"]))
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C25")), ["C27"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C26")), ["C28"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C27")), ["C29"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C28")), ["C29"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C30")), ["C31"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C31")), sorted(["C32", "C33"]))
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C32")), ["C34"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C33")), ["C34"])
        self.assertListEqual(sorted(qt.query_affected_by_relations_by_suspect_component("C34")), sorted(["C35", "C36"]))

        # expected entry components
        self.assertListEqual(qt.query_suspect_components_by_dtc("P0127"), ["C22"])
        self.assertListEqual(qt.query_suspect_components_by_dtc("P0128"), ["C30"])


if __name__ == '__main__':
    smach.set_loggers(log_info, log_debug, log_warn, log_err)  # set custom logging functions
    tf.get_logger().setLevel(logging.ERROR)
    unittest.main()
