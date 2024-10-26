#!/bin/bash

# changing the working directory to the source code directory
cd ..

CONFIG_PATH="configs/plot_loss_acc.yaml" # configurations for loss and accuracy plotting
DATA_DIR="data"
LOG_DIR="logs"
MODEL_DIR="models"
PLOT_DIR="plots"

mkdir -p "$LOG_DIR"
mkdir -p "$PLOT_DIR"


python -m src.plot.plot_loss_acc "$CONFIG_PATH" --log-dir "$LOG_DIR" --plot-dir "$PLOT_DIR"