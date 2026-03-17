#!/bin/bash

# Create a Python virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Environment setup complete."
