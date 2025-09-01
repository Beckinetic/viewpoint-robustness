import argparse
import glob
import json
import logging
import os
import pickle
import random
import sys
import warnings

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from tqdm import tqdm

from src.model.models import get_model, get_device

warnings.filterwarnings('ignore')
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
eval_backgrounds = config['data']['eval']['background']
eval_res = config['data']['eval']['res']
if config['data']['eval']['sample_size']:
    sample_size = config['data']['eval']['sample_size']
else:
    sample_size = ""
backbone = config['model']['to_eval']['backbone']
views = config['model']['to_eval']['view']
backgrounds = config['model']['to_eval']['background']
res = config['model']['to_eval']['res']
max_epoch = config['model']['to_eval']['max_epoch']


def border_cam():
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

            # initialise vessels for results
            mean_interior_cam_values = {}
            mean_border_cam_values = {}
            mean_exterior_cam_values = {}
            # save results
            save_path = os.path.join(log_dir, model_name, 'border_cam.pkl')
            results = {
                'ic': mean_interior_cam_values,
                'bc': mean_border_cam_values,
                'ec': mean_exterior_cam_values,
            }

            for eval_view in eval_views:
                mean_interior_cam_values[eval_view] = {}
                mean_border_cam_values[eval_view] = {}
                mean_exterior_cam_values[eval_view] = {}

                for eval_background in eval_backgrounds:
                    # prepare eval data
                    eval_data_folder = os.path.join(data_dir, '_'.join([eval_background, eval_view, eval_res]))
                    gt_data_folder = os.path.join(data_dir, '_'.join(['gt', eval_background, eval_view, eval_res]))

                    eval_labels_path = os.path.join(eval_data_folder, 'labels.json')
                    with open(eval_labels_path) as f:
                        eval_labels = json.load(f)
                    eval_data_paths = glob.glob(eval_data_folder + '/*.png')
                    if sample_size:
                        random.seed(42)
                        eval_data_paths = random.sample(eval_data_paths, sample_size)
                    unique_labels = sorted(set(eval_labels.values()))
                    class_map = {label: index for index, label in enumerate(unique_labels)}

                    for eval_data_path in tqdm(eval_data_paths):
                        # assessing CAM
                        img = np.array(Image.open(eval_data_path))
                        img = cv2.resize(img, (224, 224))
                        img = np.float32(img) / 255
                        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                        targets = [ClassifierOutputTarget(class_map[eval_labels[eval_data_path]])]
                        target_layers = [model.layer4]
                        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

                        # prepare ground truth image
                        eval_data_filename = os.path.basename(eval_data_path)
                        gt_image_path = os.path.join(gt_data_folder, eval_data_filename)
                        gt_image = Image.open(gt_image_path)

                        # extract alpha channel (object mask)
                        alpha_channel = gt_image.split()[-1].convert('L')
                        alpha_channel = alpha_channel.resize((224, 224), Image.LANCZOS)

                        # detect object border, interior, and exterior
                        interior_mask = np.array(alpha_channel) > 128  # Object interior
                        exterior_mask = np.array(alpha_channel) <= 128  # Background (exterior)

                        # create a border mask by blurring the alpha mask and comparing the edges
                        blurred_alpha = alpha_channel.filter(ImageFilter.GaussianBlur(radius=1))
                        border_mask = np.logical_and(
                            np.array(blurred_alpha) > 0, np.array(blurred_alpha) < 255
                        )  # border pixels

                        # remove border pixels from interior and exterior
                        interior_mask = np.logical_and(interior_mask, np.logical_not(border_mask))
                        exterior_mask = np.logical_and(exterior_mask, np.logical_not(border_mask))

                        full_cam = np.squeeze(np.array(grayscale_cams))
                        interior_cam_value = np.mean(full_cam[interior_mask].flatten())
                        exterior_cam_value = np.mean(full_cam[exterior_mask].flatten())
                        border_cam_value = np.mean(full_cam[border_mask].flatten())
                        mean_interior_cam_values[eval_view][eval_data_path] = interior_cam_value
                        mean_border_cam_values[eval_view][eval_data_path] = border_cam_value
                        mean_exterior_cam_values[eval_view][eval_data_path] = exterior_cam_value

                    logging.info(f"Eval on {eval_background}_{eval_view} complete. Saving results")
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)

            with open(save_path, 'wb') as f:
                pickle.dump(results, f)


if __name__ == '__main__':
    border_cam()
