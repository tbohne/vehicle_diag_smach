#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class StateTransition:
    """
    Represents a transition in the state machine from one state to another using a particular link.
    """

    def __init__(self, prev_state: str, curr_state: str, transition_link: str) -> None:
        """
        Inits the state transition.

        :param prev_state: previous state
        :param curr_state: current state
        :param transition_link: used transition
        """
        self.prev_state = prev_state
        self.curr_state = curr_state
        self.transition_link = transition_link

    def __str__(self) -> str:
        """
        Returns a string representation of the state transition.

        :return: string representation of state transition
        """
        return self.prev_state + " --- (" + self.transition_link + ") ---> " + self.curr_state
