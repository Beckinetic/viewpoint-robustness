import argparse
import os
import random
import sys

import yaml
from tqdm import tqdm

from src.texture_shape_bias.style_transfer import read_json_file, divide_labels, find_filenames_by_label, load_image, \
    style_transfer
from src.model.models import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Run style transfer')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = get_device()
    root = config['data']['root']
    content_labels = read_json_file(os.path.join(root, 'content/labels.json'))
    style_labels = read_json_file(os.path.join(root, 'style/labels.json'))

    num_subsets = config['params']['num_subsets']
    partitioned_labels = divide_labels(style_labels, num_subsets)

    random.seed(config['params']['seed'])

    for i, subset in enumerate(partitioned_labels):
        try:
            style_subset = partitioned_labels[i + 1]
        except:
            style_subset = partitioned_labels[0]

        print(f"Content subset: {subset}")
        print(f"Style_subset: {style_subset}")

        style_path_dict = {}

        for j, content_label in enumerate(subset):
            content_paths = find_filenames_by_label(content_labels, content_label)

            for ii, content_path in enumerate(content_paths):
                # load content
                content = load_image(content_path).to(device)
                content_name = os.path.basename(content_path).split('.')[0]

                for k, style_label in enumerate(style_subset):
                    if style_label not in style_path_dict:
                        # if style_label is not in the dictionary, randomly choose a style path
                        style_paths = find_filenames_by_label(style_labels, style_label)
                        style_path = random.choice(style_paths)
                        style_path_dict[style_label] = style_path  # Store the chosen path
                    else:
                        # if style_label is already in the dictionary, use the stored path
                        style_path = style_path_dict[style_label]
                    style_id = os.path.basename(style_path).split('.')[0].split('_')[0]

                    # load style
                    style = load_image(style_path, shape=content.shape[-2:]).to(device)

                    # style transfer and change the steps to > 2000 for better performance
                    transferred_file_name = '_'.join([content_name, str(ii), style_id + '.png'])
                    output_folder = os.path.join(root, 'output')
                    os.makedirs(output_folder, exist_ok=True)
                    style_transfer(content, style, transferred_file_name, output_folder,
                                   config['params']['num_step'])


if __name__ == '__main__':
    main()
