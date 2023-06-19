#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List


class IntermediateResults:
    """
    Represents intermediate results of the diagnosis, which are communicated to the user.
    """

    def __init__(self, status: str, required_actions: List[dict], state_log: List):
        """
        Inits the intermediate results.

        :param status: status of the diagnostic process
        :param required_actions: actions required from the user (e.g. record data at component)
        :param state_log: history of past states
        """
        self.status = status
        self.required_actions = required_actions
        self.state_log = state_log
