#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach

class PerformDataManagement(smach.State):

    def __init__(self):

        smach.State.__init__(self,
                             outcomes=['performed_data_management', 'performed_reduced_data_management'],
                             input_keys=['suggestion_list'],
                             output_keys=['suggestion_list'])

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        return "performed_data_management"
