#!/bin/bash

CONFIG_PATH="../configs/plot_mixed_loss_acc.yaml"
LOG_DIR="../logs"
PLOT_DIR="../plots"

usage() {
    echo "Usage: $0 [-c <config_path>] [-l <log_dir>] [-m <plot_dir>]"
    echo "  -c, --config       Path to the configuration file (default: configs/plot_mixed_loss_acc.yaml)"
    echo "  -l, --log-dir      Directory to load logs (default: logs/)"
    echo "  -p, --plot-dir    Directory to save plots (default: plots/)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -l|--log-dir) LOG_DIR="$2"; shift ;;
        -p|--plot-dir) PLOT_DIR="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

mkdir -p "$LOG_DIR"
mkdir -p "$PLOT_DIR"

python ../src/plot/plot_mixed_loss_acc.py "$CONFIG_PATH" --log-dir "$LOG_DIR" --plot-dir "$PLOT_DIR"