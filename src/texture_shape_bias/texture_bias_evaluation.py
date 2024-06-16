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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.models import get_device, get_model

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = get_device()


def get_cue_conflict_images(cue_conflict_folder):
    filepaths = []
    shape_labels = {}
    texture_labels = {}

    for dirpath, dirnames, files in os.walk(cue_conflict_folder):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(dirpath, file)
                filepaths.append(full_path)

                parts = os.path.basename(file).split('_')
                # Update dictionaries
                shape_id = parts[0]
                texture_id = parts[-1].split('.')[0]

                with open('../objaverse/parsed_lvis_annotations.json') as f:
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


class ContentDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

        # Automate class_map creation
        unique_labels = sorted(set(self.labels.values()))  # Sort to ensure alphabetical order
        self.class_map = {label: index for index, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')

        label = self.labels[img_path]
        class_id = torch.tensor(self.class_map[label])

        if self.transform:
            image = self.transform(image)

        return image, class_id


def evaluate_model_texture_shape_bias(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode

    # Initialize category-specific counters
    category_shape_decisions = {}
    category_texture_decisions = {}
    category_totals = {}

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

                # Update totals for each category
                if shape_label not in category_totals:
                    category_totals[shape_label] = 0
                    category_shape_decisions[shape_label] = 0
                    category_texture_decisions[shape_label] = 0

                category_totals[shape_label] += 1
                if predicted[i].item() == shape_label:
                    category_shape_decisions[shape_label] += 1
                if predicted[i].item() == texture_label:
                    category_texture_decisions[shape_label] += 1

    # Calculate the shape and texture bias for each category
    # shape_bias = {label: category_shape_decisions[label] / category_totals[label] for label in category_totals}
    # texture_bias = {label: category_texture_decisions[label] / category_totals[label] for label in category_totals}

    return category_shape_decisions, category_texture_decisions, category_totals


def evaluate_model_on_content(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing on Content Dataset"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Texture bias evaluation on Cue-Conflict Dataset')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory to load models')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # get cue-conflict dataset
    root = config['data']['root']
    cue_conflict_path = os.path.join(root, '_output')
    content_path = os.path.join(root, 'content')

    # get models and logs path
    model_dir = args.model_dir
    log_dir = args.log_dir
    model_paths = []
    for model_folder in config['model']['model_folders']:
        model_paths.append(os.path.join(model_dir, model_folder))

    # prepare cue-conflict dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_paths, shape_labels, texture_labels = get_cue_conflict_images(cue_conflict_path)
    cue_conflict_dataset = CueConflictDataset(img_paths, shape_labels, texture_labels, transform)
    cue_conflict_dataloader = torch.utils.data.DataLoader(cue_conflict_dataset, batch_size=1, shuffle=True)

    # prepare content dataset
    img_paths, labels = get_content_images(content_path)
    content_dataset = ContentDataset(img_paths, labels, transform)
    content_dataloader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=True)

    # prepare model
    model_name = config['model']['model_name']
    model, _ = get_model(model_name, pretrained=False, num_classes=32)

    # evaluate the models on content and cue-conflict dataset
    for model_path in model_paths:
        content_accuracies = []
        shape_decisions = []
        texture_decisions = []

        # go over all epochs
        epochs = len(glob.glob(os.path.join(model_path, '*.pth')))
        for epoch in tqdm(range(epochs), desc="Evaluating Models Through Epochs"):
            logging.info(f"Evaluating Model {model_path} Epoch {epoch}")
            state_dict = torch.load(os.path.join(model_path, '_'.join([model_name, 'epoch', str(epoch + 1) + '.pth'])),
                                    map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.eval()

            logging.info(f"Evaluating on content dataset")
            content_accuracy = evaluate_model_on_content(model, content_dataloader)
            logging.info(f"Evaluating on cue-conflict dataset")
            shape_decision, texture_decision = evaluate_model_texture_shape_bias(model, cue_conflict_dataloader)

            content_accuracies.append(content_accuracy)
            shape_decisions.append(shape_decision)
            texture_decisions.append(texture_decision)

        result = {'content_accuracies': content_accuracies,
                  'shape_decisions': shape_decisions,
                  'texture_decisions': texture_decisions}

        os.makedirs(os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root)), exist_ok=True)
        save_path = os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root), 'results.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main()
