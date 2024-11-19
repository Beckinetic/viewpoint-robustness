import argparse
import glob
import json
import logging
import os
import pickle
import sys

import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.model.models import get_device, get_model

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = get_device()


def get_cue_conflict_images(cue_conflict_folder):
    filepaths = []
    shape_labels = {}
    texture_labels = {}

    for dirpath, dirnames, files in os.walk(cue_conflict_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                full_path = os.path.join(dirpath, file)
                filepaths.append(full_path)

                parts = os.path.basename(file).split('_')
                # Update dictionaries
                shape_id = parts[0]
                texture_id = parts[-1].split('.')[0]

                with open('objaverse/parsed_lvis_annotations.json') as f:
                    annotations = json.load(f)
                shape_labels[full_path] = annotations[shape_id]
                texture_labels[full_path] = annotations[texture_id]

    return filepaths, shape_labels, texture_labels


def get_content_images(content_folder):
    filepaths = []
    filenames = []
    content_labels = {}
    json_file = os.path.join(content_folder, 'labels.json')
    with open(json_file, 'r') as f:
        labels = json.load(f)

    for dirpath, dirnames, files in os.walk(content_folder):
        for file in files:
            if file.endswith('.png'):
                filenames.append(file)
                full_path = os.path.join(dirpath, file)
                filepaths.append(full_path)
                content_labels[full_path] = labels[full_path]

    return filepaths, content_labels


class CueConflictDataset(Dataset):
    def __init__(self, img_paths, shape_labels, texture_labels, transform=None):
        self.img_paths = img_paths
        self.shape_labels = shape_labels
        self.texture_labels = texture_labels
        self.transform = transform

        unique_labels = sorted(set(self.shape_labels.values()))
        self.class_map = {label: index for index, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        shape_label = self.shape_labels[img_path]
        texture_label = self.texture_labels[img_path]

        shape_class_id = torch.tensor(self.class_map[shape_label])
        texture_class_id = torch.tensor(self.class_map[texture_label])

        if self.transform:
            image = self.transform(image)

        return image, shape_class_id, texture_class_id, img_path

    def get_triplet(self, idx):
        anchor_img_path = self.img_paths[idx]
        anchor_shape_label = self.shape_labels[anchor_img_path]
        anchor_texture_label = self.texture_labels[anchor_img_path]

        # find texture image
        texture_img_path = None
        for path in self.img_paths:
            if self.texture_labels[path] == anchor_texture_label and self.shape_labels[path] != anchor_shape_label:
                texture_img_path = path
                break

        # find shape image
        shape_img_path = None
        for path in self.img_paths:
            if self.shape_labels[path] == anchor_shape_label and self.texture_labels[path] != anchor_texture_label:
                shape_img_path = path
                break

        assert texture_img_path is not None, "No matching texture image found."
        assert shape_img_path is not None, "No matching shape image found."

        # load images
        anchor_image = Image.open(anchor_img_path).convert('RGB')
        texture_image = Image.open(texture_img_path).convert('RGB')
        shape_image = Image.open(shape_img_path).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            texture_image = self.transform(texture_image)
            shape_image = self.transform(shape_image)

        return anchor_image, texture_image, shape_image


def evaluate_model_shape_bias(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode

    # Initialize category-specific counters
    category_shape_decisions = {}
    category_texture_decisions = {}
    category_totals = {}

    # Initialize image-specific counters
    shape_decisions = []
    texture_decisions = []
    neither_decisions = []

    with torch.no_grad():  # Disable gradient computation
        for images, shape_labels, texture_labels, img_paths in tqdm(data_loader,
                                                                    desc="Testing on Cue-Conflict Dataset"):
            images = images.to(device)
            shape_labels = shape_labels.to(device)
            texture_labels = texture_labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, 1)

            for i in range(len(shape_labels)):
                shape_label = shape_labels[i].item()
                texture_label = texture_labels[i].item()
                image_path = img_paths[i]

                # Update totals for each category
                if shape_label not in category_totals:
                    category_totals[shape_label] = 0
                    category_shape_decisions[shape_label] = 0
                    category_texture_decisions[shape_label] = 0

                category_totals[shape_label] += 1
                if predicted[i].item() == shape_label:
                    category_shape_decisions[shape_label] += 1
                    shape_decisions.append(image_path)
                if predicted[i].item() == texture_label:
                    category_texture_decisions[shape_label] += 1
                    texture_decisions.append(image_path)
                else:
                    neither_decisions.append(image_path)

    return category_shape_decisions, category_texture_decisions, category_totals, shape_decisions, texture_decisions, neither_decisions
