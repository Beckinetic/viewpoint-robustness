import os
import json
import random
from tqdm import tqdm

import torch

import style_transfer as st


device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def find_label_by_filename(data, filename):
    return next((item['label'] for item in data if item['filename'] == filename), None)


def find_filenames_by_label(data, label):
    return [item['filename'] for item in data if item['label'] == label]


def divide_labels(data, num):
    labels = list(set(item['label'] for item in data))
    random.shuffle(labels)
    return [labels[i::num] for i in range(num)]


if __name__ == '__main__':
    root_dir = 'formal_batch_1'
    content_labels = read_json_file(os.path.join(root_dir, 'content/labels.json'))
    style_labels = read_json_file(os.path.join(root_dir, 'style/labels.json'))

    num_subsets = 4
    partitioned_labels = divide_labels(style_labels, num_subsets)

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
                content_list = os.listdir(os.path.join(root_dir, 'content'))
                if content_filename in content_list:
                    content_path = os.path.join(os.path.join(root_dir, 'content'), content_filename)
                    content = st.load_image(content_path).to(device)

                    for k, style_label in enumerate(style_subset):
                        # random choose one kind of style: I don't want to make choices tbh
                        style_filenames = find_filenames_by_label(style_labels, style_label)
                        style_list = os.listdir(os.path.join(root_dir, 'style'))
                        style_intersection = set(style_filenames).intersection(set(style_list))
                        style_filename = random.choice(list(style_intersection))

                        # load style
                        style_path = os.path.join(os.path.join(root_dir, 'style'), style_filename)
                        style = st.load_image(style_path, shape=content.shape[-2:]).to(device)

                        # style transfer and change the steps to > 2000 for better performance
                        transferred_file_name = content_label + str(ii) + '_' + style_label + '.jpg'
                        output_folder = os.path.join(os.path.join(root_dir, 'output'), content_label)
                        os.makedirs(output_folder, exist_ok=True)
                        st.style_transfer(content, style, transferred_file_name, output_folder, 200)
                else:
                    continue
