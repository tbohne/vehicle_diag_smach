#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import os
import shutil
from datetime import date
from pathlib import Path
from typing import List

from oscillogram_classification import preprocess
from py4j.java_gateway import JavaGateway

from vehicle_diag_smach.config import DUMMY_OSCILLOGRAMS, OSCI_SESSION_FILES, \
    DUMMY_ISOLATION_OSCILLOGRAM_POS, DUMMY_ISOLATION_OSCILLOGRAM_NEG1, DUMMY_ISOLATION_OSCILLOGRAM_NEG2
from vehicle_diag_smach.config import SESSION_DIR, XPS_SESSION_FILE
from vehicle_diag_smach.data_types.customer_complaint_data import CustomerComplaintData
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
        val = None
        while val != "":
            val = input("\nlocal interface impl.: simulation of mechanic providing these information..")
        return WorkshopData(4, date.today())

    def get_obd_data(self) -> OnboardDiagnosisData:
        """
        Retrieves the on-board diagnosis data required in the diagnostic process.

        :return: on-board diagnosis data
        """
        val = None
        while val != "":
            val = input("\nlocal interface impl.: sim processing OBD data..")
        obd_data = OnboardDiagnosisData(
            ['P2563'], "Mazda 3", "849357984", "453948539", "1234567890ABCDEFGHJKLMNPRSTUVWXYZ"
        )
        print(obd_data)
        return obd_data

    @staticmethod
    def create_local_dummy_oscillograms() -> None:
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
        # --> hard-coded for demo purposes - showing reasonable case
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
        val = None
        while val != "":
            val = input("\nlocal interface impl.: sim mechanic - press 'ENTER' when the recording phase is finished"
                        + " and the oscillograms are generated for " + str(components))
        self.create_local_dummy_oscillograms()
        oscillograms = []
        for comp in components:
            path = SESSION_DIR + "/" + OSCI_SESSION_FILES + "/" + comp + ".csv"
            _, voltages = preprocess.read_oscilloscope_recording(path)
            oscillograms.append(OscillogramData(voltages, comp))
        return oscillograms

    def get_customer_complaints(self) -> CustomerComplaintData:
        """
        Local implementation of customer complaint retrieval.

        :return: customer complaints
        """
        val = ""
        while val != "0" and val != "1":
            val = input("\nlocal interface impl.: starting diagnosis with [0] / without [1] customer complaints")

        if val == "0":
            # launch expert system that processes the customer complaints
            print("establish connection to customer XPS server..")
            customer_xps = JavaGateway().entry_point
            print("result of customer xps: ",
                  customer_xps.demo("../vehicle_diag_smach/" + SESSION_DIR + "/" + XPS_SESSION_FILE))
            return CustomerComplaintData(SESSION_DIR + "/" + XPS_SESSION_FILE)
        else:
            return CustomerComplaintData()

    def get_manual_judgement_for_component(self, component: str) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the specified vehicle component.

        :param component: vehicle component to get manual judgement for
        :return: true -> anomaly, false -> regular
        """
        print("local interface impl.: manual inspection of component:", component)
        val = ""
        while val not in ['0', '1']:
            val = input("\nsim mechanic - press '0' for defective component, i.e., anomaly, and '1' for no defect..")
        return val == "0"

    def get_manual_judgement_for_sensor(self) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the currently considered sensor.

        :return: true -> anomaly, false -> regular
        """
        print("no anomaly identified -- check potential sensor malfunction..")
        val = ""
        while val not in ['0', '1']:
            val = input("\npress '0' for sensor malfunction and '1' for working sensor..")
        return val == "0"
