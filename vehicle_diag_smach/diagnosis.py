#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import smach

from vehicle_diag_smach.interfaces.data_accessor import DataAccessor
from vehicle_diag_smach.interfaces.data_provider import DataProvider
from vehicle_diag_smach.interfaces.model_accessor import ModelAccessor
from vehicle_diag_smach.low_level_states.classify_components import ClassifyComponents
from vehicle_diag_smach.low_level_states.gen_artificial_instance_based_on_cc import GenArtificialInstanceBasedOnCC
from vehicle_diag_smach.low_level_states.isolate_problem_check_effective_radius import \
    IsolateProblemCheckEffectiveRadius
from vehicle_diag_smach.low_level_states.no_problem_detected_check_sensor import NoProblemDetectedCheckSensor
from vehicle_diag_smach.low_level_states.provide_diag_and_show_trace import ProvideDiagAndShowTrace
from vehicle_diag_smach.low_level_states.provide_initial_hypothesis_and_log_context import \
    ProvideInitialHypothesisAndLogContext
from vehicle_diag_smach.low_level_states.select_best_unused_error_code_instance import SelectBestUnusedErrorCodeInstance
from vehicle_diag_smach.low_level_states.suggest_suspect_components import SuggestSuspectComponents


class DiagnosisStateMachine(smach.StateMachine):
    """
    Low-level diagnosis state machine responsible for the details of the diagnostic process.
    """

    def __init__(
            self, model_accessor: ModelAccessor, data_accessor: DataAccessor, data_provider: DataProvider, kg_url: str
    ) -> None:
        """
        Initializes the low-level state machine.

        :param model_accessor: implementation of the model accessor interface
        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        super(DiagnosisStateMachine, self).__init__(
            outcomes=['refuted_hypothesis', 'diag'], input_keys=[], output_keys=['final_output']
        )
        self.model_accessor = model_accessor
        self.data_accessor = data_accessor
        self.data_provider = data_provider
        self.userdata.sm_input = []
        self.kg_url = kg_url

        with self:  # defines states and transitions of the low-level diagnosis SMACH

            self.add('SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE', SelectBestUnusedErrorCodeInstance(self.data_provider),
                     transitions={'selected_matching_instance(OBD_CC)': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'no_matching_selected_best_instance': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'no_instance': 'GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC',
                                  'no_instance_and_CC_already_used': 'NO_PROBLEM_DETECTED_CHECK_SENSOR'},
                     remapping={'selected_instance': 'sm_input', 'customer_complaints': 'sm_input'})

            self.add('ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS',
                     IsolateProblemCheckEffectiveRadius(
                         self.data_accessor, self.model_accessor, self.data_provider, self.kg_url
                     ),
                     transitions={'isolated_problem': 'PROVIDE_DIAG_AND_SHOW_TRACE',
                                  'isolated_problem_remaining_DTCs': 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE'},
                     remapping={'classified_components': 'sm_input', 'fault_paths': 'sm_input'})

            self.add('NO_PROBLEM_DETECTED_CHECK_SENSOR',
                     NoProblemDetectedCheckSensor(self.data_accessor, self.data_provider),
                     transitions={'sensor_works': 'PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT',
                                  'sensor_defective': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('GEN_ARTIFICIAL_INSTANCE_BASED_ON_CC', GenArtificialInstanceBasedOnCC(self.data_provider),
                     transitions={'generated_artificial_instance': 'SUGGEST_SUSPECT_COMPONENTS'},
                     remapping={'customer_complaints': 'sm_input', 'generated_instance': 'sm_input'})

            self.add('PROVIDE_INITIAL_HYPOTHESIS_AND_LOG_CONTEXT',
                     ProvideInitialHypothesisAndLogContext(self.data_provider, self.kg_url),
                     transitions={'no_diag': 'refuted_hypothesis'},
                     remapping={})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE', ProvideDiagAndShowTrace(self.data_provider, self.kg_url),
                     transitions={'uploaded_diag': 'diag'},
                     remapping={'diagnosis': 'sm_input', 'final_output': 'final_output'})

            self.add('CLASSIFY_COMPONENTS',
                     ClassifyComponents(self.model_accessor, self.data_accessor, self.data_provider, self.kg_url),
                     transitions={'no_anomaly_no_more_comp': 'SELECT_BEST_UNUSED_ERROR_CODE_INSTANCE',
                                  'no_anomaly': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'suggestion_list': 'sm_input', 'classified_components': 'sm_input'})

            self.add('SUGGEST_SUSPECT_COMPONENTS', SuggestSuspectComponents(self.data_provider, self.kg_url),
                     transitions={'provided_suggestions': 'CLASSIFY_COMPONENTS'},
                     remapping={'selected_instance': 'sm_input', 'generated_instance': 'sm_input',
                                'suggestion_list': 'sm_input'})
