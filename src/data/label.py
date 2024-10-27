import argparse
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


def relabel():
    def process_folder(folder_path):
        # check if there are any image files in the current folder
        if any(file.endswith(('.png', '.jpg', '.jpeg')) for file in os.listdir(folder_path) if
               os.path.isfile(os.path.join(folder_path, file))):
            create_labels(folder_path)
            print(f'Relabelled {folder_path}')

        # recursively check all subdirectories
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                process_folder(subfolder_path)

    for data_folder in data_folders:
        data_path = os.path.join(root, data_folder)
        process_folder(data_path)


if __name__ == '__main__':
    relabel()
