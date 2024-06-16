import json
import os
import random
from collections import defaultdict
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm
from .create_dataset import create_labels


def create_partition_plan(_root, num_partitions):
    """
    Create a partition plan of categories
    :param num_partitions: Number of partitions
    :param _root: root folder
    :return: a partition plan of categories
    """
    if os.path.exists(os.path.join(_root, 'labels.json')):
        label_path = os.path.join(_root, 'labels.json')
    else:
        label_path = create_labels(_root)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # count the number of images per category
    category_counts = defaultdict(int)
    for label in labels.values():
        category_counts[label] += 1

    # convert the counts to a sorted list of tuples
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # initialize groups
    groups = [[] for _ in range(num_partitions)]
    group_sizes = [0] * num_partitions

    # distribute categories into groups
    for category, count in sorted_categories:
        min_group_index = group_sizes.index(min(group_sizes))
        groups[min_group_index].append(category)
        group_sizes[min_group_index] += count

    # Prepare the final result with counts
    result = [{"categories": group, "total_images": sum(category_counts[cat] for cat in group)} for group in groups]
    print("Partition Plan Summary:")
    for i, group_info in enumerate(result):
        print(f"Group {i + 1}: Categories = {group_info['categories']}, Total Images = {group_info['total_images']}")

    return groups


def sample_by_category_and_instance(dataset, group, biased_ratio):
    instance_dict = {}
    for idx in tqdm(range(len(dataset)), desc='Sampling pictures'):
        image_path = dataset.image_paths[idx]
        _, class_id = dataset[idx]
        class_label = list(dataset.class_map.keys())[list(dataset.class_map.values()).index(class_id.item())]
        if class_label in group:
            instance_id = os.path.basename(image_path).split('_')[0]
            if instance_id not in instance_dict:
                instance_dict[instance_id] = []
            instance_dict[instance_id].append(idx)

    min_images_per_instance = min(len(images) for images in instance_dict.values())

    sampled_indices = []
    for instance_images in instance_dict.values():
        sample_size = int(len(instance_images) * biased_ratio)
        if sample_size > min_images_per_instance:
            sample_size = min_images_per_instance
        sampled_indices.extend(random.sample(instance_images, sample_size))

    print(f"Sampled {len(sampled_indices)} images for categories {group}")

    return Subset(dataset, sampled_indices)


def mix_dataset(datasets, groups, num_partitions, biased_ratio):
    if not len(datasets) == len(groups):
        raise ValueError("datasets and groups must have same number while datasets are given {} and groups are given {}".format(len(datasets), len(groups)))
    if not 0 < biased_ratio <= 1:
        raise ValueError("Biased ratio must be between 0 and 1 while given is {}".format(biased_ratio))

    sampled_datasets = []
    for i, dataset in enumerate(datasets):
        for j, group in enumerate(groups):
            if i == j:
                sampled_datasets.append(sample_by_category_and_instance(dataset, group, biased_ratio))
            else:
                sampled_datasets.append(sample_by_category_and_instance(dataset, group, (1 - biased_ratio) / (num_partitions - 1)))

    mixed_dataset = ConcatDataset(sampled_datasets)

    # Print summary of the mixed dataset
    print("Mixed Dataset Summary:")
    total_samples = sum(len(subset) for subset in sampled_datasets)
    print(f"Total number of samples in the mixed dataset: {total_samples}")
    print(f"Number of sub-datasets combined: {len(sampled_datasets)}")
    print(f"Biased ratio: {biased_ratio}")

    return mixed_dataset
