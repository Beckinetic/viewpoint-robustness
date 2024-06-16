import glob
import json
import numpy as np
import os

import torch
from PIL import Image

from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.model_selection import train_test_split


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


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths

        self.labels = labels
        unique_labels = sorted(set(self.labels.values()))  # Sort to ensure alphabetical order
        self.class_map = {label: index for index, label in enumerate(unique_labels)}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        label = self.labels[image_path]
        class_id = torch.tensor(self.class_map[label])

        if self.transform:
            image = self.transform(image)

        return image, class_id


def create_dataset(root_dir, label_path=None, transform=None):
    """
    Create dataset for training and validation sets.
    :param root_dir: root directory of images and labels files
    :param label_path: labels file path
    :param transform: transform function
    :return:
    """
    image_paths = glob.glob(os.path.join(root_dir, '*.png'))

    if label_path is None:
        if os.path.exists(os.path.join(root_dir, 'labels.json')):
            label_path = os.path.join(root_dir, 'labels.json')
        else:
            label_path = create_labels(root_dir)
    with open(label_path, 'r') as f:
        labels = json.load(f)

    dataset = CustomDataset(image_paths, labels, transform)
    return dataset


def train_val_test_split(dataset, test_size=0.2, random_state=42):
    image_paths = dataset.image_paths
    labels = dataset.labels
    label_list = [labels[img_path] for img_path in image_paths]

    train_idx, val_idx = train_test_split(
        np.arange(len(image_paths)),
        test_size=test_size,
        stratify=label_list,
        random_state=random_state
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset


def extract_indices(dataset):
    """
    Extracts indices and labels from the ConcatDataset or Subset.
    :param dataset: The dataset (either ConcatDataset or Subset)
    :return: A list of tuples containing the index in the original dataset and the corresponding label
    """
    indices_labels = []
    if isinstance(dataset, ConcatDataset):
        for ds_index, ds in enumerate(dataset.datasets):
            if isinstance(ds, Subset):
                subset_indices = ds.indices
                for idx in subset_indices:
                    _, label = ds.dataset[idx]
                    indices_labels.append((ds_index, idx, label.item()))
            else:
                for idx in range(len(ds)):
                    _, label = ds[idx]
                    indices_labels.append((ds_index, idx, label.item()))
    else:
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            indices_labels.append((0, idx, label.item()))
    return indices_labels


def train_test_split_mixed_dataset(mixed_dataset, test_size=0.2, random_state=42):
    """
    Splits the mixed dataset into training and testing sets.
    :param mixed_dataset: The mixed dataset (ConcatDataset)
    :param test_size: The proportion of the dataset to include in the test split
    :param random_state: The seed used by the random number generator
    :return: Two Subsets (train_dataset, test_dataset)
    """
    indices_labels = extract_indices(mixed_dataset)
    indices, labels = zip(*[(idx, label) for _, idx, label in indices_labels])

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_subsets = []
    test_subsets = []

    for ds_index, (ds, start_idx) in enumerate(zip(mixed_dataset.datasets, range(len(mixed_dataset.datasets)))):
        if isinstance(ds, Subset):
            subset_indices = ds.indices
            train_subset_indices = [idx for idx in train_indices if start_idx <= idx < start_idx + len(subset_indices)]
            test_subset_indices = [idx for idx in test_indices if start_idx <= idx < start_idx + len(subset_indices)]

            train_subsets.append(Subset(ds.dataset, train_subset_indices))
            test_subsets.append(Subset(ds.dataset, test_subset_indices))
        else:
            train_subset_indices = [idx for idx in train_indices if start_idx <= idx < start_idx + len(ds)]
            test_subset_indices = [idx for idx in test_indices if start_idx <= idx < start_idx + len(ds)]

            train_subsets.append(Subset(ds, train_subset_indices))
            test_subsets.append(Subset(ds, test_subset_indices))

    train_dataset = ConcatDataset(train_subsets)
    test_dataset = ConcatDataset(test_subsets)

    return train_dataset, test_dataset

# Example usage:
# mixed_dataset = mix_dataset(datasets, groups, num_partitions, biased_ratio)
# train_dataset, test_dataset = train_test_split_mixed_dataset(mixed_dataset)



# def merge_log_files(files):
#     """
#     Merge Unity log files into one file.
#     :param files: log files generated by Unity
#     :return: merged log file
#     """
#     merged_data = []
#
#     for file in files:
#         with open(file, 'r') as f:
#             data = json.load(f)
#             for item in data["configurations"]:
#                 merged_data.append({
#                     "screenshotID": item["screenshotID"],
#                     "classOfObject": item["classOfObject"]
#                 })
#
#     return merged_data
