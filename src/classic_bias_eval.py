import argparse
import logging
import os
import pickle
import sys

import torch
import yaml
from torchvision import transforms

from model.models import get_model, get_device
from texture_shape_bias.texture_bias_evaluation import get_cue_conflict_images, CueConflictDataset, get_content_images, \
    ContentDataset, evaluate_model_on_content, evaluate_model_texture_shape_bias


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = get_device()


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
    if config['data']['pasted']:
        cue_conflict_path = os.path.join(root, 'pasted_output')
    else:
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

    # evaluate the models on content and cue-conflict dataset
    for model_path in model_paths:
        content_accuracies = []
        shape_decisions = []
        texture_decisions = []
        total_decisions = []

        # load the state dict of the last epoch
        last_epoch = config['training']['last_epoch']
        logging.info(f'Loading epoch: {last_epoch}')
        state_dict = torch.load(os.path.join(model_path, '_'.join([model_name, 'epoch', str(last_epoch) + '.pth'])),
                                map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logging.info(f"Evaluating on content dataset")
        content_accuracy = evaluate_model_on_content(model, content_dataloader, device)
        logging.info(f"Evaluating on cue-conflict dataset")
        shape_decision, texture_decision, total_decision = evaluate_model_texture_shape_bias(model, cue_conflict_dataloader, device)

        content_accuracies.append(content_accuracy)
        shape_decisions.append(shape_decision)
        texture_decisions.append(texture_decision)
        total_decisions.append(total_decision)

        result = {'content_accuracies': content_accuracies,
                  'shape_decisions': shape_decisions,
                  'texture_decisions': texture_decisions,
                  'total_decisions': total_decisions}

        os.makedirs(os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root)), exist_ok=True)
        if config['data']['pasted']:
            save_file_name = 'shape_bias_pasted.pkl'
        else:
            save_file_name = 'shape_bias.pkl'
        save_path = os.path.join(log_dir, os.path.basename(model_path), os.path.basename(root), save_file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    main()
