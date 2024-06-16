import argparse
import glob
import logging
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision import transforms

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.models import get_model, get_device
from texture_bias_evaluation import get_cue_conflict_images, CueConflictDataset, \
    get_content_images, ContentDataset, evaluate_model_on_content

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = get_device()


class ModelWithPenultimateFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(ModelWithPenultimateFeatureExtractor, self).__init__()
        self.model = model
        self.penultimate_layer = list(model.children())[-2]  # Assuming penultimate layer is the second to last

    def forward(self, x):
        for name, module in self.model.named_children():
            x = module(x)
            if module == self.penultimate_layer:
                break
        return x


def triplet_bias_eval(model, dataloader, dataset, device):
    model.eval()
    total = 0
    shape_decisions = 0
    texture_decisions = 0
    with torch.no_grad():  # disable gradient computation
        for anchor_img, shape_labels, texture_labels, anchor_img_path in tqdm(dataloader,
                                                                              desc="Testing on Cue-Conflict Dataset"):
            # get the anchor index
            anchor_idx = dataset.img_paths.index(anchor_img_path[0])

            # fetch the triplet images
            anchor_img, texture_img, shape_img = dataset.get_triplet(anchor_idx)

            # move to device
            anchor_img = anchor_img.to(device).unsqueeze(0)
            texture_img = texture_img.to(device).unsqueeze(0)
            shape_img = shape_img.to(device).unsqueeze(0)

            # extract features
            anchor_features = model(anchor_img)
            texture_features = model(texture_img)
            shape_features = model(shape_img)

            # compute cosine distances
            cos_dist_texture = F.cosine_similarity(anchor_features, texture_features)
            cos_dist_shape = F.cosine_similarity(anchor_features, shape_features)

            # decision counting
            if cos_dist_texture > cos_dist_shape:
                texture_decisions += 1
            else:
                shape_decisions += 1

            # update total
            total += len(shape_labels)

    return shape_decisions / total, texture_decisions / total


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
    cue_conflict_path = os.path.join(root, 'output')
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
    model.to(device)

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
            content_accuracy = evaluate_model_on_content(model, content_dataloader, device)
            logging.info(f"Evaluating on cue-conflict dataset")
            shape_decision, texture_decision = triplet_bias_eval(model, cue_conflict_dataloader, cue_conflict_dataset,
                                                                 device)

            content_accuracies.append(content_accuracy)
            shape_decisions.append(shape_decision)
            texture_decisions.append(texture_decision)

        result = {'content_accuracies': content_accuracies, 'shape_decisions': shape_decisions,
                  'texture_decisions': texture_decisions}

        os.makedirs(os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root)), exist_ok=True)
        save_path = os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root), 'triplet_results.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main()
