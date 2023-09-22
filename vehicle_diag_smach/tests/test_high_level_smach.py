#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import logging
import unittest

import smach
import tensorflow as tf

from vehicle_diag_smach.high_level_smach import VehicleDiagnosisStateMachine
from vehicle_diag_smach.io.local_model_accessor import LocalModelAccessor
from vehicle_diag_smach.io.test_data_accessor import TestDataAccessor
from vehicle_diag_smach.io.test_data_provider import TestDataProvider
from vehicle_diag_smach.util import log_info, log_debug, log_warn, log_err


class TestHighLevelStateMachine(unittest.TestCase):
    """
    Multiple test scenarios to ensure basic functionality of the entire diagnostic process.
    """

    def test_hosted_knowledge_graph(self) -> None:
        """
        Tests whether the correct (expected) KG is hosted for the unit tests:
        https://github.com/tbohne/obd_ontology/tree/main/knowledge_base/unit_test_kg.nt
        """
        # TODO: Add asserts for assumed expert knowledge (KG queries).

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
