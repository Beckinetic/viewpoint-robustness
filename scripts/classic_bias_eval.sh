#!/bin/bash

CONFIG_PATH="../configs/classic_bias_eval.yaml"
LOG_DIR="../logs"
MODEL_DIR="../models"

usage() {
    echo "Usage: $0 [-c <config_path>] [-l <log_dir>] [-m <model_dir>]"
    echo "  -c, --config       Path to the configuration file (default: configs/default_texture_bias_evaluation.yaml)"
    echo "  -l, --log-dir      Directory to save logs (default: logs/)"
    echo "  -m, --model-dir    Directory to save models (default: models/)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -l|--log-dir) LOG_DIR="$2"; shift ;;
        -m|--model-dir) MODEL_DIR="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

python ../src/classic_bias_eval.py "$CONFIG_PATH" --log-dir "$LOG_DIR" --model-dir "$MODEL_DIR"