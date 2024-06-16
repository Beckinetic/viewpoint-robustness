#!/bin/bash

CONFIG_PATH="../configs/rename.yaml"
DATASET_PATH="../data"

usage() {
    echo "Usage: $0 [-c <config_path>] [-d <dataset>]"
    echo "  -c, --config       Path to the configuration file (default: configs/rename.yaml)"
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

python ../src/data/rename.py "$CONFIG_PATH" --dataset "$DATASET_PATH"
