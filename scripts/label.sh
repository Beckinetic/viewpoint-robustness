#!/bin/bash

# changing the working directory to the source code directory
cd ..

CONFIG_PATH="configs/label.yaml" # configurations for loss and accuracy plotting
DATA_DIR="data"
LOG_DIR="logs"
MODEL_DIR="models"
PLOT_DIR="plots"

mkdir -p "$LOG_DIR"
mkdir -p "$PLOT_DIR"


python -m src.data.label "$CONFIG_PATH" --data-dir "$DATA_DIR"