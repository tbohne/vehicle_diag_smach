#!/bin/bash

/usr/bin/java -Xms4g -Xmx8g -Dlog4j2.formatMsgNoLookups=true -XX:-UseGCOverheadLimit -XX:+UseParallelGC -jar fuseki-server.jar --loc=/code/fuseki &
python3 /code/vehicle_diag_smach/high_level_smach.py
