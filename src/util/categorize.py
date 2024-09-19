import json
import os
import shutil

from src.data.create_dataset import create_labels


def categorize(root):
    label_path = create_labels(root)
    with open(label_path, 'r') as file:
        labels = json.load(file)

    for file_path, label in labels.items():
        category_dir = os.path.join(os.path.dirname(file_path), label)
        os.makedirs(category_dir, exist_ok=True)

        # Move the file to the new category directory
        shutil.move(file_path, os.path.join(category_dir, os.path.basename(file_path)))

    print('Done')