#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os

import smach
from termcolor import colored


class PerformSynchronizedSensorRecordings(smach.State):
    """
    State in the low-level SMACH that represents situations in which the synchronized sensor recordings are performed
    at the suggested suspect components.
    """

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['processed_sync_sensor_data'],
                             input_keys=['suggestion_list'],
                             output_keys=[''])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS' state.

        :param userdata: input of state
        :return: outcome of the state ("processed_sync_sensor_data")
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("PERFORM_SYNCHRONIZED_SENSOR_RECORDINGS", "yellow", "on_grey", ["bold"]), "state..")
        print("############################################\n")

        components_to_be_recorded = [k for k, v in userdata.suggestion_list.items() if v]
        components_to_be_manually_verified = [k for k, v in userdata.suggestion_list.items() if not v]
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be verified manually:", components_to_be_manually_verified)
        print("------------------------------------------")

        # TODO: perform manual verification of components and let mechanic enter result + communicate
        #       anomalies further for fault isolation

        print(colored("\nperform synchronized sensor recordings at:", "green", "on_grey", ["bold"]))
        for comp in components_to_be_recorded:
            print(colored("- " + comp, "green", "on_grey", ["bold"]))

        val = None
        while val != "":
            val = input("\npress 'ENTER' when the recording phase is finished and the oscillograms are generated..")

        return "processed_sync_sensor_data"
