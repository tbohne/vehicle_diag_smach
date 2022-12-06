#!/bin/bash

# create a new session "session_one"
tmux new-session -d -s session_one

# run fuseki server
tmux send-keys -t session_one "./apache-jena-fuseki-4.6.1/fuseki-server" Enter

# split terminal window horizontally
tmux split-window -h -t session_one

# run smach in the second window
tmux send-keys -t session_one "python3 /code/vehicle_diag_smach/high_level_smach.py" Enter

# attach to session
tmux attach -t session_one
