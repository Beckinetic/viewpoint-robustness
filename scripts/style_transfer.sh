#!/bin/bash

cd ..

CONFIG_PATH="configs/style_transfer.yaml"

python -m src.create_cue_conflict "$CONFIG_PATH"