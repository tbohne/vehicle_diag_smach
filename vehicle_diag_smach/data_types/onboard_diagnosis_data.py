#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class OnboardDiagnosisData:
    """
    Represents onboard diagnosis data, which is communicated to the state machine.
    """

    def __init__(self, dtc_list: list[str], model: str, hsn: str, tsn: str, vin: str):
        """
        Inits the onboard diagnosis data.

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
