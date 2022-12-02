#!/bin/bash

./apache-jena-fuseki-4.6.1/fuseki-server &
python3 /code/vehicle_diag_smach/high_level_smach.py
