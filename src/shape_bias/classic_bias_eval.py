import argparse
import logging
import os
import pickle
import sys

import torch
import yaml
from torchvision import transforms

from src.data.create_dataset import CustomDataset
from src.model.models import get_model, get_device
from src.shape_bias.shape_bias_eval import get_cue_conflict_images, CueConflictDataset, get_content_images, evaluate_model_shape_bias
from src.model.test import test_model

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
backbone = config['model']['backbone']
views = config['model']['to_eval']['view']
res = config['model']['to_eval']['res']
backgrounds = config['model']['to_eval']['background']
max_epoch = config['model']['to_eval']['max_epoch']


def classic_bias_eval():
    for view in views:
        for background in backgrounds:
            # load model
            model_name = '_'.join([background, view, res])
            model_folder = os.path.join(model_dir, model_name)
            model_path = os.path.join(model_folder, '_'.join([backbone, 'epoch', str(max_epoch) + '.pth']))

            model, _ = get_model(backbone, pretrained=False, num_classes=num_classes)

            # initialise test results
            content_accuracies = {}
            shape_decisions = {}
            texture_decisions = {}
            total_decisions = {}

            image_shape_decisions = {}
            image_texture_decisions = {}
            image_neither_decisions = {}

            # get cue-conflict dataset
            for eval_view in eval_views:
                for suffix in suffixes:
                    if suffix:
                        cue_conflict_data_path = os.path.join(data_dir, '_'.join(['cue_conflict', eval_view, suffix]),
                                                              'output')
                    else:
                        cue_conflict_data_path = os.path.join(data_dir, '_'.join(['cue_conflict', eval_view]),
                                                              'output')
                    content_data_path = os.path.join(data_dir, '_'.join(['cue_conflict', eval_view]), 'content')

                    # prepare cue-conflict dataset
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_paths, shape_labels, texture_labels = get_cue_conflict_images(cue_conflict_data_path)
                    cue_conflict_dataset = CueConflictDataset(img_paths, shape_labels, texture_labels, transform)
                    cue_conflict_dataloader = torch.utils.data.DataLoader(cue_conflict_dataset, batch_size=1, shuffle=True)

                    # prepare content dataset
                    img_paths, labels = get_content_images(content_data_path)
                    content_dataset = CustomDataset(img_paths, labels, transform)
                    content_dataloader = torch.utils.data.DataLoader(content_dataset, batch_size=1, shuffle=True)

                    # load the state dict of the last epoch
                    state_dict = torch.load(model_path, map_location=torch.device(device))
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()

                    logging.info(f"Evaluating on content dataset")
                    content_accuracy = test_model(model, content_dataloader)
                    logging.info(f"Evaluating on cue-conflict {eval_view} dataset")
                    shape_decision, texture_decision, total_decision, image_shape_decision,\
                        image_texture_decision, image_neither_decision = evaluate_model_shape_bias(model,
                                                                                                     cue_conflict_dataloader,
                                                                                                     device)
                    if suffix:
                        key = '_'.join([eval_view, suffix])
                    else:
                        key = eval_view
                    content_accuracies[key] = content_accuracy
                    shape_decisions[key] = shape_decision
                    texture_decisions[key] = texture_decision
                    total_decisions[key] = total_decision
                    image_shape_decisions[key] = image_shape_decision
                    image_texture_decisions[key] = image_texture_decision
                    image_neither_decisions[key] = image_neither_decision

            # save results
            result = {'content_accuracies': content_accuracies,
                      'shape_decisions': shape_decisions,
                      'texture_decisions': texture_decisions,
                      'total_decisions': total_decisions}

            decision_filter = {
                'isd': image_shape_decisions,
                'itd': image_texture_decisions,
                'ind': image_neither_decisions,
            }
            save_path = os.path.join(log_dir, model_name, 'shape_bias.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)
            filter_save_path = os.path.join(log_dir, model_name, 'decision_filter.pkl')
            with open(filter_save_path, 'wb') as f:
                pickle.dump(decision_filter, f)


if __name__ == '__main__':
    classic_bias_eval()
