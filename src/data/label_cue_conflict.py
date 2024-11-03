import argparse
import glob
import json
import os

import yaml

from src.data.create_dataset import create_labels

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('config', type=str, help='Path to the configuration file')
parser.add_argument('--data-dir', type=str, default='data/', help='Directory to fetch the data')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

root = args.data_dir
data_folders = config['data_folders']


def relabel_cue_conflict():
    for folder in data_folders:
        image_paths = glob.glob(os.path.join(root, folder, 'output', '*.jpg'))
        with open('objaverse/parsed_lvis_annotations.json') as f:
            annotations = json.load(f)

        shape_labels = {}
        texture_labels = {}
        for image_path in image_paths:
            shape_image_id = image_path.split('/')[-1].split('_')[0].split('.')[0]
            texture_image_id = image_path.split('/')[-1].split('_')[-1].split('.')[0]
            shape_labels[image_path] = annotations[shape_image_id]
            texture_labels[image_path] = annotations[texture_image_id]

        with open(os.path.join(root, folder, 'output', 'shape_labels.json'), 'w') as f:
            json.dump(shape_labels, f, indent=4)

        with open(os.path.join(root, folder, 'output', 'texture_labels.json'), 'w') as f:
            json.dump(texture_labels, f, indent=4)

        print(f"Labelled {folder}")


if __name__ == '__main__':
    relabel_cue_conflict()
