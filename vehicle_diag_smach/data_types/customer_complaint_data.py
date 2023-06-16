#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import xml.etree.ElementTree as EleTree


class CustomerComplaintData:
    """
    Represents customer complaints submitted from the hub UI to smach.
    """

    def __init__(self, xps_session_file: str = ""):
        """
        Initializes the customer complaint data.

        :param xps_session_file: path to the customer complaint expert system session file
        """
        self.root = EleTree.parse(xps_session_file).getroot() if xps_session_file != "" else ""

    def print_all_info(self, element: str = "", indent: int = 0):
        """
        Recursively prints all the information of the XPS session.

        :param element: element of the XPS tree to print info for
        :param indent: level of indentation
        """
        if element == "":
            element = self.root
        print(" " * indent + f"{element.tag}: {element.attrib}")
        if element.text and element.text.strip():
            print(" " * (indent + 2) + f"Text: {element.text.strip()}")
        for child in element:
            self.print_all_info(child, indent + 2)
