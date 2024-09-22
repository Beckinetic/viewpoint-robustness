import argparse
import glob
import json
import logging
import os
import pickle
import sys

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.create_dataset import create_labels
from src.model.models import get_model, get_device

device = get_device()
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


class RobustnessTestDataset(Dataset):
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


def evaluation(model, data_loader, _device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing on current dataset"):
            images = images.to(_device)
            labels = labels.to(_device)

            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Robustness Test')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save models')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # get robustness dataset
    root = config['data']['root']
    viewpoint = config['data']['viewpoint']
    root = os.path.join(root, viewpoint)

    distortion_types = config['distortion_types']
    severities = range(1, 6)

    # get models to test
    model_dir = args.model_dir
    model_name = config['model']['model_name']
    model_folders = config['model']['model_folders']
    epochs = range(30, 31)

    # save results dir
    log_dir = args.log_dir

    # model settings
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model, _ = get_model(model_name, pretrained=False, num_classes=32)

    for model_folder in model_folders:
        # create vessels to store results
        content_acc = {}
        distortion_acc = {}

        for epoch in epochs:
            logging.info(f"{model_folder}, Epoch {epoch}")
            model_path = os.path.join(model_dir, model_folder, f'resnet18_epoch_{epoch}.pth')
            state_dict = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # evaluate content accuracy
            logging.info(f"Evaluating on content dataset")
            content_image_paths = glob.glob(os.path.join(root, 'content', '*.png'))
            content_labels_path = create_labels(os.path.join(root, 'content'))
            with open(content_labels_path, 'r') as f:
                content_labels = json.load(f)
            content_dataset = RobustnessTestDataset(content_image_paths, content_labels, transform)
            content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=False)
            content_acc[str(epoch)] = evaluation(model, content_loader, device)
            logging.info(f"Content Accuracy on epoch {epoch}: {content_acc[str(epoch)]}")

            # evaluate distortion accuracy
            logging.info(f"Evaluating on distortion dataset")
            distortion_acc[str(epoch)] = {}
            for distortion_type in distortion_types:
                distortion_acc[str(epoch)][distortion_type] = {}
                for severity in severities:
                    logging.info(f"{distortion_type}, severity: {severity}")
                    distortion_image_paths = glob.glob(os.path.join(root, distortion_type, str(severity), '*.png'))
                    distortion_labels_path = create_labels(os.path.join(root, distortion_type, str(severity)))
                    with open(distortion_labels_path, 'r') as f:
                        distortion_labels = json.load(f)
                    distortion_dataset = RobustnessTestDataset(distortion_image_paths, distortion_labels, transform)
                    distortion_loader = torch.utils.data.DataLoader(distortion_dataset, batch_size=1, shuffle=False)
                    distortion_acc[str(epoch)][distortion_type][str(severity)] = evaluation(model, distortion_loader,
                                                                                            device)
                    logging.info(
                        f"Distortion accuracy on epoch {epoch}, {distortion_type}, severity: {severity}: {distortion_acc[str(epoch)][distortion_type][str(severity)]}")

            # path to save results
            log_model_epoch_folder = os.path.join(log_dir, model_folder, 'robustness', str(epoch), viewpoint)
            save_path_content_acc = os.path.join(log_model_epoch_folder, 'content_acc.pkl')
            save_path_distortion_acc = os.path.join(log_model_epoch_folder, 'distortion_acc.pkl')

            os.makedirs(log_model_epoch_folder, exist_ok=True)
            with open(save_path_content_acc, 'wb') as f:
                pickle.dump(content_acc, f)

            with open(save_path_distortion_acc, 'wb') as f:
                pickle.dump(distortion_acc, f)


if __name__ == '__main__':
    main()
