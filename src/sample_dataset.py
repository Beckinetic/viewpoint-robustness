import argparse
import os

import yaml

from data.sample_dataset import sample_dataset_by_instance, sample_dataset_by_category
from data.create_dataset import create_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, help='Path to the datasets')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    output = os.path.join(args.dataset, config['sample']['output'])
    os.makedirs(output, exist_ok=True)

    for background in config['data']['background']:
        for view in config['data']['view']:
            for res in config['data']['res']:
                root = os.path.join(args.dataset, '_'.join([background, view, res]))
                sample_dataset_by_category(root, config['sample']['sample_num'], 42, True, output)

    create_labels(output)
    print('Done')
