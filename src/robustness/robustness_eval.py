import argparse
import glob
import json
import logging
import os
import pickle
import sys

import torch
import yaml
from torch.utils.data import Dataset
from torchvision import transforms

from src.model.test import test_model
from src.data.create_dataset import create_labels, CustomDataset
from src.model.models import get_model, get_device

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
device = get_device()

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('config', type=str, help='Path to the configuration file')
parser.add_argument('--data-dir', type=str, default='data/', help='Directory to fetch the data')
parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save models')
args = parser.parse_args()

data_dir = args.data_dir
log_dir = args.log_dir
model_dir = args.model_dir

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

num_classes = config['data']['eval']['num_classes']
eval_views = config['data']['eval']['view']
suffixes = ['']
if config['data']['eval']['suffix'] is not None:
    suffixes.extend(config['data']['eval']['suffix'])
distortion_types = config['data']['eval']['distortion_types']
backbone = config['model']['backbone']
views = config['model']['to_eval']['view']
res = config['model']['to_eval']['res']
backgrounds = config['model']['to_eval']['background']
max_epoch = config['model']['to_eval']['max_epoch']
severities = range(1, 6)


def robustness_eval():
    for view in views:
        for background in backgrounds:
            # load model
            model_name = '_'.join([background, view, res])
            model_folder = os.path.join(model_dir, model_name)
            model_path = os.path.join(model_folder, '_'.join([backbone, 'epoch', str(max_epoch) + '.pth']))
            general_backbone_name = backbone.split('_')[0]
            model, _ = get_model(general_backbone_name, pretrained=False, num_classes=num_classes)
            logging.info(f"{model_folder}, Epoch {max_epoch}")
            state_dict = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # transform settings
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # initialise result vessels
            distortion_acc = {}
            distortion_loss = {}
            content_acc = {}
            content_loss = {}

            # distorted data path
            for eval_view in eval_views:
                for suffix in suffixes:
                    if suffix:
                        distorted_data_path = os.path.join(data_dir, '_'.join(['robustness', eval_view, suffix]))
                        key = '_'.join([eval_view, suffix])
                    else:
                        distorted_data_path = os.path.join(data_dir, '_'.join(['robustness', eval_view]))
                        key = eval_view
                    # evaluate content accuracy
                    logging.info(f"Evaluating on content dataset")
                    content_image_paths = glob.glob(os.path.join(distorted_data_path, 'content', '*.png'))
                    content_labels_path = create_labels(os.path.join(distorted_data_path, 'content'))
                    with open(content_labels_path, 'r') as f:
                        content_labels = json.load(f)
                    content_dataset = CustomDataset(content_image_paths, content_labels, transform)
                    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=False)
                    content_loss[key], content_acc[key] = test_model(model, content_loader)
                    logging.info(f"Content Loss on epoch {max_epoch}: {content_loss[key]}")
                    logging.info(f"Content Accuracy on epoch {max_epoch}: {content_acc[key]}")

                    # evaluate distortion accuracy
                    logging.info(f"Evaluating on distortion dataset")
                    distortion_acc[key] = {}
                    distortion_loss[key] = {}
                    for distortion_type in distortion_types:
                        distortion_acc[key][distortion_type] = {}
                        distortion_loss[key][distortion_type] = {}
                        for severity in severities:
                            logging.info(f"{distortion_type}, severity: {severity}")
                            distortion_image_paths = glob.glob(os.path.join(distorted_data_path, distortion_type,
                                                                            str(severity), '*.png'))
                            distortion_labels_path = create_labels(os.path.join(distorted_data_path, distortion_type,
                                                                                str(severity)))
                            with open(distortion_labels_path, 'r') as f:
                                distortion_labels = json.load(f)
                            distortion_dataset = CustomDataset(distortion_image_paths, distortion_labels, transform)
                            distortion_loader = torch.utils.data.DataLoader(distortion_dataset, batch_size=1, shuffle=False)
                            (distortion_loss[key][distortion_type][str(severity)],
                             distortion_acc[key][distortion_type][str(severity)]) = test_model(model, distortion_loader)
                            logging.info(
                                f"Distortion accuracy on epoch {max_epoch}, {distortion_type}, severity: {severity}: "
                                f"{distortion_acc[key][distortion_type][str(severity)]}")

            # save results
            results = {'da': distortion_acc,
                       'dl': distortion_loss,
                       'ca': content_acc,
                       'cl': content_loss}

            save_path = os.path.join(log_dir, model_name, 'robustness.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    robustness_eval()
