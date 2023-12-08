#!/bin/bash


# Create a virtual environment
python3 -m venv venv

chmod +x Newdata_prediction.sh

chmod +x requirments.txt 

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
# pip install -r results/requirements.txt

pip3 install -r requirments.txt

   
# Run your Python script
python3 new_domain_prediction.py

# Deactivate the virtual environment
deactivate


# Delete the virtual environment
rm -rf venv