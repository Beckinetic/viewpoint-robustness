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


def main():
    for data_folder in data_folders:
        data_path = os.path.join(root, data_folder)
        create_labels(data_path)
        print('Done')


if __name__ == '__main__':
    main()
