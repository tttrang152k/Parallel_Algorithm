#!/bin/bash

# Define the Python script to be run
PYTHON_SCRIPT="facebookPerformanceAnalysisParallel.py"

# Run the script with 2, 4, and 8 processes using mpiexec
echo "Running with 2 processes..."
mpiexec -n 2 python3 $PYTHON_SCRIPT

echo "Running with 4 processes..."
mpiexec -n 4 python3 $PYTHON_SCRIPT

echo "Running with 8 processes..."
mpiexec -n 8 python3 $PYTHON_SCRIPT
