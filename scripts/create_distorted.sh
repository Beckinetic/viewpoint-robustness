#!/bin/bash

CONFIG_PATH="../configs/create_distorted.yaml"

usage() {
    echo "Usage: $0 [-c <config_path>]"
    echo "  -c, --config       Path to the configuration file (default: configs/create_distorted.yaml)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

python ../src/robustness/create_distorted.py "$CONFIG_PATH"