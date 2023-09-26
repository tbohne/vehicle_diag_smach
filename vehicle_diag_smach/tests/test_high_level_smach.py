#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import logging
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


if __name__ == '__main__':
    smach.set_loggers(log_info, log_debug, log_warn, log_err)  # set custom logging functions
    tf.get_logger().setLevel(logging.ERROR)
    unittest.main()
