#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os
import shutil
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from oscillogram_classification import preprocess

from vehicle_diag_smach.config import DUMMY_OSCILLOGRAMS, OSCI_SESSION_FILES, \
    DUMMY_ISOLATION_OSCILLOGRAM_POS, DUMMY_ISOLATION_OSCILLOGRAM_NEG1, DUMMY_ISOLATION_OSCILLOGRAM_NEG2
from vehicle_diag_smach.config import SESSION_DIR
from vehicle_diag_smach.data_types.customer_complaint_data import CustomerComplaintData
from vehicle_diag_smach.data_types.onboard_diagnosis_data import OnboardDiagnosisData
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.workshop_data import WorkshopData
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor


class TestDataAccessor(DataAccessor):
    """
    Implementation of the data accessor interface used in unit test scenarios.
    """

    def __init__(self, test_scenario: int = 0) -> None:
        """
        Initializes the test data accessor.

        :param test_scenario: determines the test scenario
        """
        self.test_scenario = test_scenario

    def get_workshop_info(self) -> WorkshopData:
        """
        Retrieves the workshop metadata required in the diagnostic process.

        :return: workshop metadata
        """
        return WorkshopData(4, date.today())

    def get_obd_data(self) -> OnboardDiagnosisData:
        """
        Retrieves the on-board diagnosis data required in the diagnostic process.

        :return: on-board diagnosis data
        """
        obd_data = OnboardDiagnosisData(['P0125'], "TestCar", "123", "456", "ABC")
        if self.test_scenario == 0:
            obd_data = OnboardDiagnosisData(['P0125'], "TestCar", "123", "456", "ABC")
        elif self.test_scenario == 1:
            obd_data = OnboardDiagnosisData(['P0126'], "TestCar", "123", "456", "ABC")
        elif self.test_scenario == 2:
            obd_data = OnboardDiagnosisData(['P0127'], "TestCar", "123", "456", "ABC")
        elif self.test_scenario == 3:
            obd_data = OnboardDiagnosisData(['P0128'], "TestCar", "123", "456", "ABC")
        elif self.test_scenario == 4:
            obd_data = OnboardDiagnosisData(['P0127', 'P0128'], "TestCar", "123", "456", "ABC")
        return obd_data

    def create_local_dummy_oscillograms(self) -> None:
        """
        Creates local dummy oscillograms to be used in the classification states.
        """
        # set up directory for osci classification
        osci_session_dir = SESSION_DIR + "/" + OSCI_SESSION_FILES + "/"
        if not os.path.exists(osci_session_dir):
            os.makedirs(osci_session_dir)

        # create dummy oscillograms in '/session_files'
        for path in Path(DUMMY_OSCILLOGRAMS).rglob('*.csv'):
            src = str(path)
            shutil.copy(src, osci_session_dir + str(src.split("/")[-1]))

        if self.test_scenario == 0:
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C1" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG2, osci_session_dir + "C3" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C5" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C7" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C9" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C11" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C13" + ".csv")
        elif self.test_scenario == 1:
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C15" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C17" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C19" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C21" + ".csv")
        elif self.test_scenario == 2:
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C23" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C25" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C27" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C29" + ".csv")
        elif self.test_scenario == 3:
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C31" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C33" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C35" + ".csv")
        elif self.test_scenario == 4:
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C23" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C25" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C27" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C29" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C31" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "C33" + ".csv")
            shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "C35" + ".csv")

    def get_oscillograms_by_components(self, components: List[str]) -> List[OscillogramData]:
        """
        Retrieves the oscillogram data for the specified components.

        :param components: components to retrieve oscillograms for
        :return: oscillogram data for each component
        """
        self.create_local_dummy_oscillograms()
        oscillograms = []
        for comp in components:
            path = SESSION_DIR + "/" + OSCI_SESSION_FILES + "/" + comp + ".csv"
            _, voltages = preprocess.read_oscilloscope_recording(path)
            voltages = [pd.DataFrame(voltages)]
            oscillograms.append(OscillogramData(voltages, comp))
        return oscillograms

    def get_customer_complaints(self) -> CustomerComplaintData:
        """
        Local implementation of customer complaint retrieval.

        :return: customer complaints
        """
        return CustomerComplaintData()

    def get_manual_judgement_for_component(self, component: str) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the specified vehicle component.

        :param component: vehicle component to get manual judgement for
        :return: true -> anomaly, false -> regular
        """
        if ((self.test_scenario == 0 and int(component[-1]) in [2, 4])
                or (self.test_scenario == 1 and int(component[1:]) in [14, 18, 20])
                or ((self.test_scenario == 2 or self.test_scenario == 4) and int(component[1:]) in [22, 24, 26, 28])
                or ((self.test_scenario == 3 or self.test_scenario == 4) and int(component[1:]) in [30, 32, 34, 36])):
            return True
        return False

    def get_manual_judgement_for_sensor(self) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the currently considered sensor.

        :return: true -> anomaly, false -> regular
        """
        return False
