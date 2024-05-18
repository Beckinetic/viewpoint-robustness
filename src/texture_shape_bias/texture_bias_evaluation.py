import json
import logging
import os
import pickle
import re
import sys

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def get_cue_conflict_images(cue_conflict_folder):
    filepaths = []
    shape_labels = {}
    texture_labels = {}

    for dirpath, dirnames, files in os.walk(cue_conflict_folder):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(dirpath, file)
                filepaths.append(full_path)

                filename_without_extension = os.path.splitext(file)[0]

                pattern = re.compile(r'(\D+)(\d*)_(.*)\.jpg')
                match = pattern.match(file)
                if match:
                    shape_label, instance_number, texture_label = match.groups()

                # Update dictionaries
                shape_labels[full_path] = shape_label
                texture_labels[full_path] = texture_label

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

        for item in labels:
            if item['filename'] in filenames:
                content_labels[os.path.join(content_folder, item['filename'])] = item['label']

    return filepaths, content_labels


class CueConflictDataset(Dataset):
    def __init__(self, img_paths, shape_labels, texture_labels, transform=None):
        """
        img_paths: List with all the image paths.
        shape_labels: Dictionary mapping image paths to their shape labels.
        texture_labels: Dictionary mapping image paths to their texture labels.
        transform: Optional transform to be applied on a sample.
        """
        self.img_paths = img_paths
        self.shape_labels = shape_labels
        self.texture_labels = texture_labels
        self.transform = transform

        # Automate class_map creation
        unique_labels = sorted(set(self.shape_labels.values()))  # Sort to ensure consistent order
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

        return image, shape_class_id, texture_class_id


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


def evaluate_model_texutre_shape_bias(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    shape_decisions = 0
    texture_decisions = 0
    with torch.no_grad():  # Disable gradient computation
        for images, shape_labels, texture_labels in tqdm(data_loader, desc="Testing on Cue-Conflict Dataset"):
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += len(shape_labels)
            shape_decisions += (predicted == shape_labels).sum().item()
            texture_decisions += (predicted == texture_labels).sum().item()
    return shape_decisions / total, texture_decisions / total


def evaluate_model_on_content(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing on Content Dataset"):
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    data_folder = '../../data'
    cue_conflict_path = os.path.join(data_folder, 'cue_conflict_rf_meadow')
    content_path = os.path.join(data_folder, 'content_rf_meadow')
    model_main_folder = '../../saved_models'
    model_folders = []
    for dirpath, dirnames, files in os.walk(model_main_folder):
        for dirname in dirnames:
            model_folders.append(os.path.join(dirpath, dirname))

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

    # prepare model (resnet18 backbone)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 32)

    # evaluate the models on content and cue-conflict dataset
    for model_folder in model_folders:
        content_accuracies = []
        shape_decisions = []
        texture_decisions = []
        # go over all epochs
        for epoch in tqdm(range(30), desc="Evaluating Models Through Epochs"):
            logging.info(f"Evaluating Model {model_folder} Epoch {epoch}")
            state_dict = torch.load(os.path.join(model_folder, str(epoch) + '.pth'), map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.eval()

            logging.info(f"Evaluating on content dataset")
            content_accuracy = evaluate_model_on_content(model, content_dataloader)
            logging.info(f"Evaluating on cue-conflict dataset")
            shape_decision, texture_decision = evaluate_model_texutre_shape_bias(model, cue_conflict_dataloader)

            content_accuracies.append(content_accuracy)
            shape_decisions.append(shape_decision)
            texture_decisions.append(texture_decision)

        result = {'content_accuracies': content_accuracies,
                  'shape_decisions': shape_decisions,
                  'texture_decisions': texture_decisions}

        save_path = os.path.join(model_folder, 'cue_conflict_results.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
