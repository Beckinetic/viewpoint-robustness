import glob
import json
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm

from src.data.create_dataset import create_labels


def sample_dataset_by_category(root, sample_num, seed, save, output):
    # read or create labels.json
    if os.path.exists(os.path.join(root, 'labels.json')):
        label_path = os.path.join(root, 'labels.json')
    else:
        label_path = create_labels(root)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # organize images by category
    category_dict = defaultdict(list)
    for image_path, category in labels.items():
        category_dict[category].append(image_path)

    # sample images per category and copy to the output directory
    sampled_labels = {}
    random.seed(seed)
    for category, image_paths in tqdm(category_dict.items()):
        sampled_images = random.sample(image_paths, min(sample_num, len(image_paths)))
        for image_path in sampled_images:
            if save:
                # copy the image to the output directory
                output_path = os.path.join(output, os.path.basename(image_path))
                shutil.copy(image_path, output_path)
                sampled_labels[output_path] = category
            else:
                sampled_labels[image_path] = category

            # Read metadata from the original logs.json
            original_metadata_path = os.path.join(os.path.dirname(image_path), 'logs.json')
            with open(original_metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
                image_metadata = next((item for item in metadata['configurations'] if
                                       item['screenshotID'] == os.path.basename(image_path)), None)

                if image_metadata:
                    # Read or create the metadata file in the output directory
                    output_metadata_path = os.path.join(output, 'logs.json')
                    if os.path.exists(output_metadata_path):
                        with open(output_metadata_path, 'r') as out_metadata_file:
                            output_metadata = json.load(out_metadata_file)
                    else:
                        output_metadata = {"configurations": []}

                    # Add or update the metadata in the output directory
                    output_metadata['configurations'].append(image_metadata)

                    # Save the updated metadata
                    with open(output_metadata_path, 'w') as out_metadata_file:
                        json.dump(output_metadata, out_metadata_file, indent=4)

    return sampled_labels


def sample_dataset_by_instance(_root, sample_num, seed, _save, _output):
    # read or create labels.json
    if os.path.exists(os.path.join(_root, 'labels.json')):
        label_path = os.path.join(_root, 'labels.json')
    else:
        label_path = create_labels(_root)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # organize images by instance
    instance_dict = defaultdict(list)
    for image_path, _ in labels.items():
        instance_dict[os.path.basename(image_path).split('_')[0]].append(image_path)

    # sample images per instance and copy to the output directory
    sampled_labels = {}
    random.seed(seed)
    for instance, image_paths in tqdm(instance_dict.items(), desc='Sampling instances'):
        sampled_images = random.sample(image_paths, min(sample_num, len(image_paths)))
        for image_path in sampled_images:
            if _save:
                # copy the image to the output directory
                output_path = os.path.join(_output, os.path.basename(image_path))
                shutil.copy(image_path, output_path)
                sampled_labels[output_path] = labels[image_path]
            else:
                sampled_labels[image_path] = labels[image_path]

            # Read metadata from the original logs.json
            original_metadata_path = os.path.join(os.path.dirname(image_path), 'logs.json')
            with open(original_metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
                image_metadata = next((item for item in metadata['configurations'] if
                                       item['screenshotID'] == os.path.basename(image_path)), None)

                if image_metadata:
                    # Read or create the metadata file in the output directory
                    output_metadata_path = os.path.join(_output, 'logs.json')
                    if os.path.exists(output_metadata_path):
                        with open(output_metadata_path, 'r') as out_metadata_file:
                            output_metadata = json.load(out_metadata_file)
                    else:
                        output_metadata = {"configurations": []}

                    # Add or update the metadata in the output directory
                    output_metadata['configurations'].append(image_metadata)

                    # Save the updated metadata
                    with open(output_metadata_path, 'w') as out_metadata_file:
                        json.dump(output_metadata, out_metadata_file, indent=4)

    print(f'Sampled images: {len(sampled_labels)} from {_root}')
    return sampled_labels
