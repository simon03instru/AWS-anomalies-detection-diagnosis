#!/bin/bash
# Setup script for Weather Anomaly Monitor

echo "Setting up Weather Anomaly Monitor..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
#mkdir -p logs output checkpoints/dki_staklim dataset/dataset_fix

echo "Setup complete!"
echo "To activate the virtual environment: source venv/bin/activate"
echo "To run the monitor: python main.py"