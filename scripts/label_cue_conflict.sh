#!/bin/bash

# changing the working directory to the source code directory
cd ..

CONFIG_PATH="configs/label_cue_conflict.yaml" # configurations for loss and accuracy plotting
DATA_DIR="data"
LOG_DIR="logs"
MODEL_DIR="models"
PLOT_DIR="plots"

mkdir -p "$LOG_DIR"
mkdir -p "$PLOT_DIR"


python -m src.data.label_cue_conflict "$CONFIG_PATH" --data-dir "$DATA_DIR"