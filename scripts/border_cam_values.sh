#!/bin/bash

CONFIG_PATH="../configs/border_cam_values.yaml"
LOG_DIR="../logs"
MODEL_DIR="../models"

usage() {
    echo "Usage: $0 [-c <config_path>] [-l <log_dir>] [-m <model_dir>]"
    echo "  -c, --config       Path to the configuration file (default: configs/border_cam_values.yaml)"
    echo "  -l, --log-dir      Directory to save logs (default: logs/)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -l|--log-dir) LOG_DIR="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

mkdir -p "$LOG_DIR"

python ../src/activation/border_cam_values.py "$CONFIG_PATH" --log-dir "$LOG_DIR"