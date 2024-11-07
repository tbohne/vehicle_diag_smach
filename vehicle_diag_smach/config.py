#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

KG_URL = "http://127.0.0.1:3030"
SESSION_DIR = "session_files"
OSCI_SESSION_FILES = "oscillograms"
SELECTED_OSCILLOGRAMS = "selected_oscillograms"
XPS_SESSION_FILE = "xps_session.xml"
SUGGESTION_SESSION_FILE = "session_suggestions.json"
CLASSIFICATION_LOG_FILE = "classifications.json"
HISTORICAL_INFO_FILE = "historical_info.txt"
OBD_INFO_FILE = "obd_info.json"
CC_TMP_FILE = "cc_tmp.json"
DTC_TMP_FILE = "dtc_tmp.json"
FAULT_PATH_TMP_FILE = "fault_paths_tmp.json"
SUS_COMP_TMP_FILE = "sus_comp_tmp.json"
TRAINED_MODEL_POOL = "res/trained_model_pool/"

DUMMY_OSCILLOGRAMS = "res/dummy_oscillograms/"
DUMMY_ISOLATION_OSCILLOGRAM_POS = "res/dummy_isolation_oscillogram/dummy_isolation_POS.csv"
DUMMY_ISOLATION_OSCILLOGRAM_NEG1 = "res/dummy_isolation_oscillogram/dummy_isolation_NEG1.csv"
DUMMY_ISOLATION_OSCILLOGRAM_NEG2 = "res/dummy_isolation_oscillogram/dummy_isolation_NEG2.csv"

# univariate keras demo data -- make sure to host corresponding KG
# FINAL_DEMO_TEST_SAMPLES = "res/prev_demonstrator_res/univariate_signals/"
# FINAL_DEMO_MODELS = "res/prev_demonstrator_res/keras_models/"
# SEED = 42
# VEHICLE_DATA = "res/prev_demonstrator_res/vehicle_info.json"
# WORKSHOP_DATA = "res/prev_demonstrator_res/workshop_info.json"
# ONLY_NEG_SAMPLES = False

# multivariate torch demo data -- make sure to host corresponding KG
FINAL_DEMO_TEST_SAMPLES = "res/final_demonstrator_res/multivariate_signals/"
FINAL_DEMO_MODELS = "res/final_demonstrator_res/final_demo_models/"
SEED = 42
VEHICLE_DATA = "res/final_demonstrator_res/vehicle_info.json"
WORKSHOP_DATA = "res/final_demonstrator_res/workshop_info.json"
ONLY_NEG_SAMPLES = True
