import argparse
import glob
import json
import logging
import os
import pickle
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
backbone = config['model']['to_eval']['backbone']
views = config['model']['to_eval']['view']
backgrounds = config['model']['to_eval']['background']
res = config['model']['to_eval']['res']
max_epoch = config['model']['to_eval']['max_epoch']


def find_gt_image(image_filename, gt_folders):
    for gt_folder in gt_folders:
        gt_image_path = os.path.join(gt_folder, image_filename)
        if os.path.exists(gt_image_path):
            return gt_image_path

    logging.warning(f'Could not find ground truth image for {image_filename}')
    return None


def border_cam_cue_conflict():
    for view in views:
        for background in backgrounds:
            # load model
            model_name = '_'.join([background, view, res])
            model_folder = os.path.join(model_dir, model_name)
            model_path = os.path.join(model_folder, '_'.join([backbone, 'epoch', str(max_epoch) + '.pth']))
            model, _ = get_model(backbone, pretrained=False, num_classes=num_classes)
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
            mean_interior_cam_values_shape = {}
            mean_border_cam_values_shape = {}
            mean_exterior_cam_values_shape = {}
            mean_interior_cam_values_texture = {}
            mean_border_cam_values_texture = {}
            mean_exterior_cam_values_texture = {}
            save_path_shape = os.path.join(log_dir, model_name, 'border_cam_cue_conflict_shape.pkl')
            save_path_texture = os.path.join(log_dir, model_name, 'border_cam_cue_conflict_texture.pkl')
            results_shape = {
                'ic': mean_interior_cam_values_shape,
                'bc': mean_border_cam_values_shape,
                'ec': mean_exterior_cam_values_shape,
            }
            results_texture = {
                'ic': mean_interior_cam_values_texture,
                'bc': mean_border_cam_values_texture,
                'ec': mean_exterior_cam_values_texture,
            }

            for eval_view in eval_views:
                mean_interior_cam_values_shape[eval_view] = {}
                mean_border_cam_values_shape[eval_view] = {}
                mean_exterior_cam_values_shape[eval_view] = {}
                mean_interior_cam_values_texture[eval_view] = {}
                mean_border_cam_values_texture[eval_view] = {}
                mean_exterior_cam_values_texture[eval_view] = {}

                # prepare eval data
                eval_data_folder = os.path.join(data_dir, '_'.join(['cue_conflict', eval_view]), 'output')

                # prepare gt data
                gt_data_folders = []
                for eval_background in eval_backgrounds:
                    gt_data_folder = os.path.join(data_dir, '_'.join(['gt', eval_background, eval_view, eval_res]))
                    gt_data_folders.append(gt_data_folder)

                eval_shape_labels_path = os.path.join(eval_data_folder, 'shape_labels.json')
                eval_texture_labels_path = os.path.join(eval_data_folder, 'texture_labels.json')
                with open(eval_shape_labels_path) as f:
                    shape_labels = json.load(f)
                with open(eval_texture_labels_path) as f:
                    texture_labels = json.load(f)

                eval_data_paths = glob.glob(eval_data_folder + '/*.jpg')
                unique_labels = sorted(set(shape_labels.values()))
                class_map = {label: index for index, label in enumerate(unique_labels)}

                for eval_data_path in tqdm(eval_data_paths):
                    # assessing CAM
                    img = np.array(Image.open(eval_data_path))
                    img = cv2.resize(img, (224, 224))
                    img = np.float32(img) / 255
                    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                    targets_shape = [ClassifierOutputTarget(class_map[shape_labels[eval_data_path]])]
                    targets_texture = [ClassifierOutputTarget(class_map[texture_labels[eval_data_path]])]
                    target_layers = [model.layer4]
                    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                        shape_cams = cam(input_tensor=input_tensor, targets=targets_shape)
                        texture_cams = cam(input_tensor=input_tensor, targets=targets_texture)

                    # prepare ground truth image
                    eval_data_filename = os.path.basename(eval_data_path)
                    gt_filename = '_'.join(eval_data_filename.split('_')[0:3]) + '.png'
                    gt_image_path = find_gt_image(gt_filename, gt_data_folders)
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

                    full_cam_shape = np.squeeze(np.array(shape_cams))
                    full_cam_texture = np.squeeze(np.array(texture_cams))

                    interior_cam_value_shape = np.mean(full_cam_shape[interior_mask].flatten())
                    exterior_cam_value_shape = np.mean(full_cam_shape[exterior_mask].flatten())
                    border_cam_value_shape = np.mean(full_cam_shape[border_mask].flatten())
                    interior_cam_value_texture = np.mean(full_cam_texture[interior_mask].flatten())
                    exterior_cam_value_texture = np.mean(full_cam_texture[exterior_mask].flatten())
                    border_cam_value_texture = np.mean(full_cam_texture[border_mask].flatten())

                    mean_interior_cam_values_shape[eval_view][eval_data_path] = interior_cam_value_shape
                    mean_border_cam_values_shape[eval_view][eval_data_path] = border_cam_value_shape
                    mean_exterior_cam_values_shape[eval_view][eval_data_path] = exterior_cam_value_shape
                    mean_interior_cam_values_texture[eval_view][eval_data_path] = interior_cam_value_texture
                    mean_border_cam_values_texture[eval_view][eval_data_path] = border_cam_value_texture
                    mean_exterior_cam_values_texture[eval_view][eval_data_path] = exterior_cam_value_texture

                logging.info(f"Eval on cue_conflict_{eval_view} complete. Saving results")
                with open(save_path_shape, 'wb') as f:
                    pickle.dump(results_shape, f)
                with open(save_path_texture, 'wb') as f:
                    pickle.dump(results_texture, f)


if __name__ == '__main__':
    border_cam_cue_conflict()