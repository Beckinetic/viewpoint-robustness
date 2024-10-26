#!/bin/bash

cd ..

CONFIG_PATH="configs/sample_dataset.yaml"
DATASET_PATH="data"

python -m src.util.sample_dataset "$CONFIG_PATH" --dataset "$DATASET_PATH"
