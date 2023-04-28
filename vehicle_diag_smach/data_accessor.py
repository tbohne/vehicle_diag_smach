#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod

from vehicle_diag_smach.data_types.onboard_diagnosis_data import OnboardDiagnosisData
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.workshop_data import WorkshopData


class DataAccessor(ABC):
    """
    Interface that defines the state machine's access to measurements and diagnosis-relevant case data.
    """

    @abstractmethod
    def get_workshop_info(self) -> WorkshopData:
        pass

    @abstractmethod
    def get_obd_data(self) -> OnboardDiagnosisData:
        pass

    @abstractmethod
    def get_oscillogram_by_component(self, component: str) -> OscillogramData:
        pass
