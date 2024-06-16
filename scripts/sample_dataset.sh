#!/bin/bash

CONFIG_PATH="../configs/sample_dataset.yaml"
DATASET_PATH="../data"

usage() {
    echo "Usage: $0 [-c <config_path>] [-d <dataset>]"
    echo "  -c, --config       Path to the configuration file (default: configs/sample_dataset.yaml)"
    echo "  -d, --dataset      Directory to load data (default: data/)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -d|--dataset) DATASET_PATH="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

python ../src/sample_dataset.py "$CONFIG_PATH" --dataset "$DATASET_PATH"
