#!/bin/bash

CONFIG_PATH="../configs/style_transfer.yaml"

usage() {
    echo "Usage: $0 [-c <config_path>]"
    echo "  -c, --config       Path to the configuration file (default: configs/default_style_transfer.yaml)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

python ../src/create_cue_conflict.py "$CONFIG_PATH"