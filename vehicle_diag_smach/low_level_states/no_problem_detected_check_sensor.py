#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored

from vehicle_diag_smach.interfaces.data_accessor import DataAccessor


class NoProblemDetectedCheckSensor(smach.State):
    """
    State in the low-level SMACH that represents situations in which no actual anomaly was detected and the indirect
    conclusion of a potential sensor malfunction is provided. This conclusion should be verified / refuted in this
    state.
    """

    def __init__(self, data_accessor: DataAccessor):
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        """
        smach.State.__init__(self,
                             outcomes=['sensor_works', 'sensor_defective'],
                             input_keys=[''],
                             output_keys=[''])
        self.data_accessor = data_accessor

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'NO_PROBLEM_DETECTED_CHECK_SENSOR' state.

        :param userdata: input of state
        :return: outcome of the state ("sensor_works" | "sensor_defective")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("NO_PROBLEM_DETECTED_CHECK_SENSOR", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################")

        anomaly = self.data_accessor.get_manual_judgement_for_sensor()
        if anomaly:
            return "sensor_defective"
        return "sensor_works"
