#! /usr/bin/bash
# This script is used to run the project pipeline.

# It is assumed that the necessary Python environment is already set up.
# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the main EDA script
if [[ $1 == "eda" ]]; then
    echo "Running EDA"
    python ./eda/eda.py
fi