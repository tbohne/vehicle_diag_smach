#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os
import shutil
from datetime import date
from pathlib import Path
from typing import List

from oscillogram_classification import preprocess

from vehicle_diag_smach.config import DUMMY_OSCILLOGRAMS, SESSION_DIR, OSCI_SESSION_FILES, \
    DUMMY_ISOLATION_OSCILLOGRAM_POS, DUMMY_ISOLATION_OSCILLOGRAM_NEG1, DUMMY_ISOLATION_OSCILLOGRAM_NEG2
from vehicle_diag_smach.data_types.onboard_diagnosis_data import OnboardDiagnosisData
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.workshop_data import WorkshopData
from vehicle_diag_smach.interfaces.data_accessor import DataAccessor


class LocalDataAccessor(DataAccessor):
    """
    Implementation of the data accessor interface using local files and dummy data for I/O.
    """

    def __init__(self):
        pass

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
        print("\nprocessing OBD data..")
        obd_data = OnboardDiagnosisData(
            ['P2563'], "Mazda 3", "849357984", "453948539", "1234567890ABCDEFGHJKLMNPRSTUVWXYZ"
        )
        print(obd_data)
        return obd_data

    @staticmethod
    def create_local_dummy_oscillograms():
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

        # create dummy oscillograms for fault isolation
        # TODO: hard-coded for demo purposes - showing reasonable case
        shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "Ladedruck-Regelventil" + ".csv")
        shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG2, osci_session_dir + "Ladedrucksteller-Positionssensor" + ".csv")
        shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "VTG-Abgasturbolader" + ".csv")
        shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_POS, osci_session_dir + "Motor-Steuergerät" + ".csv")
        shutil.copy(DUMMY_ISOLATION_OSCILLOGRAM_NEG1, osci_session_dir + "Ladedruck-Magnetventil" + ".csv")

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
            oscillograms.append(OscillogramData(voltages, comp))
        return oscillograms
