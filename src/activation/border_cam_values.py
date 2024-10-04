import argparse
import logging
import os
import pickle
import sys

import yaml


logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run picture distortion')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, help='Path to the logs directory')
    return parser.parse_args()


def find_gt_image(distorted_image_filename, gt_folders):
    for gt_folder in gt_folders:
        gt_image_path = os.path.join(gt_folder, distorted_image_filename)
        if os.path.exists(gt_image_path):
            return gt_image_path

    logging.warning(f'Could not find ground truth image for {distorted_image_filename}')


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    log_dir = args.log_dir

    model_folders = config['model']['model_folders']
    epochs = [30]

    views = config['data']['views']
    for model_folder in model_folders:
        for epoch in epochs:
            for view in views:
                cams_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'cams.pkl')
                with open(cams_path, 'rb') as f:
                    cams = pickle.load(f)

