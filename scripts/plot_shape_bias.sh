#!/bin/bash

# changing the working directory to the source code directory
# shellcheck disable=SC2164
cd ..

CONFIG_PATH="configs/plot_shape_bias.yaml" # configurations for loss and accuracy plotting
DATA_DIR="data"
LOG_DIR="logs"
MODEL_DIR="models"
PLOT_DIR="plots"

mkdir -p "$LOG_DIR"
mkdir -p "$PLOT_DIR"

python -m src.plot.plot_shape_bias "$CONFIG_PATH" --log-dir "$LOG_DIR" --plot-dir "$PLOT_DIR"