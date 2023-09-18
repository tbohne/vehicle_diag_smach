#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import List

from vehicle_diag_smach.data_types.customer_complaint_data import CustomerComplaintData
from vehicle_diag_smach.data_types.onboard_diagnosis_data import OnboardDiagnosisData
from vehicle_diag_smach.data_types.oscillogram_data import OscillogramData
from vehicle_diag_smach.data_types.workshop_data import WorkshopData


class DataAccessor(ABC):
    """
    Interface that defines the state machine's access to measurements and diagnosis-relevant case data.
    """

    @abstractmethod
    def get_workshop_info(self) -> WorkshopData:
        """
        Retrieves the workshop metadata required in the diagnostic process.

        :return: workshop metadata
        """
        pass

    @abstractmethod
    def get_obd_data(self) -> OnboardDiagnosisData:
        """
        Retrieves the on-board diagnosis data required in the diagnostic process.

        :return: on-board diagnosis data
        """
        pass

    @abstractmethod
    def get_oscillograms_by_components(self, components: List[str]) -> List[OscillogramData]:
        """
        Retrieves the oscillogram data for the specified components.

        :param components: components to retrieve oscillograms for
        :return: oscillogram data for each component
        """
        pass

    @abstractmethod
    def get_customer_complaints(self) -> CustomerComplaintData:
        """
        Retrieves customer complaints for the vehicle to be diagnosed.
        If there are no customer complaints, an empty object should be returned, i.e., CustomerComplaintData().

        :return: customer complaints
        """
        pass

    @abstractmethod
    def get_manual_judgement_for_component(self, component: str) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the specified vehicle component.

        :param component: vehicle component to get manual judgement for
        :return: true -> anomaly, false -> regular
        """
        pass

    @abstractmethod
    def get_manual_judgement_for_sensor(self) -> bool:
        """
        Retrieves a manual judgement by the mechanic for the currently considered sensor.

        :return: true -> anomaly, false -> regular
        """
        pass
