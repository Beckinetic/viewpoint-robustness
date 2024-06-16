import glob
import json
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm


def create_labels(root_dir):
    """
    Creates labels file
    :param root_dir: root directory of dataset
    :return: labels path
    """
    image_paths = glob.glob(root_dir + '/*.png')
    with open('../objaverse/parsed_lvis_annotations.json') as f:
        annotations = json.load(f)

    labels = {}
    for image_path in image_paths:
        image_id = image_path.split('/')[-1].split('_')[0]
        labels[image_path] = annotations[image_id]

    with open(os.path.join(root_dir, 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=4)

    return os.path.join(root_dir, 'labels.json')


def sample_dataset_by_category(_root, sample_num, seed, _save, _output):
    """
    Sample dataset for training and testing.
    :param _save: if True, save the sampled dataset
    :param _root: root directory of dataset
    :param _output: _output directory of sampled dataset
    :param sample_num: number of pictures per category
    :return:
    """
    # read or create labels.json
    if os.path.exists(os.path.join(_root, 'labels.json')):
        label_path = os.path.join(_root, 'labels.json')
    else:
        label_path = create_labels(_root)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # organize images by category
    category_dict = defaultdict(list)
    for image_path, category in labels.items():
        category_dict[category].append(image_path)

    # sample images per category and copy to the _output directory
    sampled_labels = {}
    random.seed(seed)
    for category, image_paths in tqdm(category_dict.items()):
        sampled_images = random.sample(image_paths, min(sample_num, len(image_paths)))
        for image_path in sampled_images:
            if _save:
                # copy the image to the _output directory
                output_path = os.path.join(_output, os.path.basename(image_path))
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

    return sampled_labels


def sample_dataset_by_instance(_root, sample_num, seed, _save, _output):
    """
    Sample dataset for training and testing by instances
    :param _root: root directory of dataset
    :param sample_num: number of pictures per category
    :param _save: if True, save the sampled dataset
    :param _output: _output directory of sampled dataset
    :return:
    """
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

    # sample images per instance and copy to the _output directory
    sampled_labels = {}
    random.seed(seed)
    for instance, image_paths in tqdm(instance_dict.items(), desc='Sampling instances'):
        sampled_images = random.sample(image_paths, min(sample_num, len(image_paths)))
        for image_path in sampled_images:
            if _save:
                # copy the image to the _output directory
                output_path = os.path.join(_output, os.path.basename(image_path))
                shutil.copy(image_path, output_path)
                sampled_labels[output_path] = labels[image_path]
            else:
                sampled_labels[image_path] = labels[image_path]

    print(f'Sampled images: {len(sampled_labels)} from {_root}')
    return sampled_labels
