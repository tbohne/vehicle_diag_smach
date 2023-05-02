#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored


class NoProblemDetectedCheckSensor(smach.State):
    """
    State in the low-level SMACH that represents situations in which no actual anomaly was detected and the indirect
    conclusion of a potential sensor malfunction is provided. This conclusion should be verified / refuted in this
    state.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['sensor_works', 'sensor_defective'],
                             input_keys=[''],
                             output_keys=[''])

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

        print("no anomaly identified -- check potential sensor malfunction..")

        val = ""
        while val not in ['0', '1']:
            val = input("\npress '0' for sensor malfunction and '1' for working sensor..")

        if val == "0":
            return "sensor_defective"
        return "sensor_works"
