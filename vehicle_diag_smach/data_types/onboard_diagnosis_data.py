#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List, Dict


class OnboardDiagnosisData:
    """
    Represents on-board diagnosis data, which is communicated to the state machine.
    """

    def __init__(self, dtc_list: List[str], model: str, hsn: str, tsn: str, vin: str) -> None:
        """
        Inits the on-board diagnosis data.

        :param dtc_list: list of diagnostic trouble codes
        :param model: model of the vehicle to be diagnosed
        :param hsn: manufacturer key number
        :param tsn: type key number
        :param vin: vehicle identification number
        """
        self.dtc_list = dtc_list
        self.model = model
        self.hsn = hsn
        self.tsn = tsn
        self.vin = vin

    def get_json_representation(self) -> Dict[str]:
        """
        Returns a JSON representation of the OBD data.

        :return: JSON representation of OBD data
        """
        return {
            "dtc_list": self.dtc_list,
            "model": self.model,
            "hsn": self.hsn,
            "tsn": self.tsn,
            "vin": self.vin
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the OBD data.

        :return: string representation of OBD data
        """
        return f"DTC list: {self.dtc_list},\nmodel: {self.model},\nHSN: {self.hsn},\nTSN: {self.tsn},\nVIN: {self.vin}"
