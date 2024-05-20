import argparse
import os
import random

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

    for i, subset in tqdm(enumerate(partitioned_labels)):
        try:
            style_subset = partitioned_labels[i + 1]
        except:
            style_subset = partitioned_labels[0]

        print(f"Content subset: {subset}")
        print(f"Style_subset: {style_subset}")

        for j, content_label in enumerate(subset):
            content_filenames = find_filenames_by_label(content_labels, content_label)

            for ii, content_filename in enumerate(content_filenames):
                # load content
                content_list = os.listdir(os.path.join(root, 'content'))
                if content_filename in content_list:
                    content_path = os.path.join(os.path.join(root, 'content'), content_filename)
                    content = load_image(content_path).to(device)

                    for k, style_label in enumerate(style_subset):
                        # random choose one kind of style
                        style_filenames = find_filenames_by_label(style_labels, style_label)
                        style_list = os.listdir(os.path.join(root, 'style'))
                        style_intersection = set(style_filenames).intersection(set(style_list))
                        style_filename = random.choice(list(style_intersection))

                        # load style
                        style_path = os.path.join(os.path.join(root, 'style'), style_filename)
                        style = load_image(style_path, shape=content.shape[-2:]).to(device)

                        # style transfer and change the steps to > 2000 for better performance
                        transferred_file_name = content_label + str(ii) + '_' + style_label + str(k) + '.jpg'
                        output_folder = os.path.join(os.path.join(root, 'output'), content_label)
                        os.makedirs(output_folder, exist_ok=True)
                        style_transfer(content, style, transferred_file_name, output_folder,
                                       config['params']['num_step'])
                else:
                    continue


if __name__ == '__main__':
    main()
