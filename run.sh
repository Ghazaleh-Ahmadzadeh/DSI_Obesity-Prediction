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

# Run data preprocessing
if [[ $1 == "preproc" ]]; then
    echo "Running Data Preprocessing"
    python ./data-preprocessing/data-preprocessing.py
fi

# Run model training
if [[ $1 == "train" ]]; then
    echo "Running Model Training"
    python ./model-training/model-training.py
fi