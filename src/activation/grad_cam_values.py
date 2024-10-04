import argparse
import glob
import json
import logging
import os
import pickle
import sys
import warnings

import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
from sklearn.metrics import auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.models import get_model, get_device
from src.data.create_dataset import create_labels

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


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

    log_dir = args.log_dir
    model_dir = args.model_dir

    model_folders = config['model']['model_folders']
    model_name = config['model']['model_name']
    epochs = [30]

    device = get_device()

    data_root = config['data']['root']
    backgrounds = config['data']['backgrounds']
    views = config['data']['views']
    resolution = config['data']['resolution']

    for model_folder in model_folders:
        model_path_all_epochs = os.path.join(model_dir, model_folder)
        activation_path_all_epochs = os.path.join(log_dir, model_folder, 'activations')
        for epoch in epochs:
            # specify the model path
            model_path = os.path.join(model_path_all_epochs, f'{model_name}_epoch_{epoch}.pth')
            # specify and create the result folder
            activation_path = os.path.join(activation_path_all_epochs, str(epoch))
            os.makedirs(activation_path, exist_ok=True)

            # load model to test
            model, _ = get_model(model_name, pretrained=False, num_classes=32)
            model.to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            for view in views:
                logging.info(f'Evaluating model: {model_folder}; Epoch : {epoch}; Test data view: {view}')
                activation_view_path = os.path.join(activation_path, view)
                os.makedirs(activation_view_path, exist_ok=True)

                cams = {}
                ious = {}
                roc_aucs = {}

                for background in backgrounds:
                    ood_data_folder = os.path.join(data_root, f"ood_{background}_{view}_{resolution}")
                    gt_data_folder = os.path.join(data_root, f"gt_ood_{background}_{view}_{resolution}")
                    ood_images_labels_path = create_labels(ood_data_folder)
                    with open(ood_images_labels_path) as f:
                        ood_images_labels = json.load(f)
                    ood_images_paths = glob.glob(ood_data_folder + '/*.png')
                    # gt_images_paths = glob.glob(gt_data_folder + '/*.png')
                    unique_labels = sorted(set(ood_images_labels.values()))
                    class_map = {label: index for index, label in enumerate(unique_labels)}

                    for ood_images_path in tqdm(ood_images_paths):
                        img = np.array(Image.open(ood_images_path))
                        img = cv2.resize(img, (224, 224))
                        img = np.float32(img) / 255
                        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                        targets = [ClassifierOutputTarget(class_map[ood_images_labels[ood_images_path]])]
                        target_layers = [model.layer4]
                        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

                        # prepare ground truth image
                        ood_image_filename = os.path.basename(ood_images_path)
                        gt_image_path = os.path.join(gt_data_folder, ood_image_filename)
                        gt_image = Image.open(gt_image_path)
                        gt_alpha_channel = gt_image.split()[-1].convert('L')
                        gt_mask = np.array(gt_alpha_channel)
                        gt_mask = cv2.resize(gt_mask, (224, 224))
                        gt_mask = (gt_mask > 0).astype(np.uint8)

                        cams[ood_image_filename] = grayscale_cams

                        cam_flat = grayscale_cams.flatten()
                        gt_flat = gt_mask.flatten()

                        # compute IoU at a specific threshold (e.g., 0.5)
                        threshold_iou = 0.5
                        cam_binary_iou = (cam_flat >= threshold_iou).astype(np.uint8)
                        intersection = np.logical_and(cam_binary_iou, gt_flat)
                        union = np.logical_or(cam_binary_iou, gt_flat)
                        iou = np.sum(intersection) / np.sum(union)
                        ious[ood_image_filename] = iou

                        # initialize thresholds for ROC curve
                        thresholds = np.linspace(0, 1, 101)

                        # initialize variables to accumulate TP, FP, FN, TN across all images
                        tp_all = np.zeros(len(thresholds))
                        fp_all = np.zeros(len(thresholds))
                        fn_all = np.zeros(len(thresholds))
                        tn_all = np.zeros(len(thresholds))

                        for i, thresh in enumerate(thresholds):
                            cam_binary = (cam_flat >= thresh).astype(np.uint8)

                            tp = np.sum(np.logical_and(cam_binary == 1, gt_flat == 1))
                            fp = np.sum(np.logical_and(cam_binary == 1, gt_flat == 0))
                            fn = np.sum(np.logical_and(cam_binary == 0, gt_flat == 1))
                            tn = np.sum(np.logical_and(cam_binary == 0, gt_flat == 0))

                            tp_all[i] += tp
                            fp_all[i] += fp
                            fn_all[i] += fn
                            tn_all[i] += tn

                        # compute True Positive Rate (TPR) and False Positive Rate (FPR) at each threshold
                        tpr = tp_all / (tp_all + fn_all)
                        fpr = fp_all / (fp_all + tn_all)

                        # handle any NaN values resulting from division by zero
                        tpr = np.nan_to_num(tpr)
                        fpr = np.nan_to_num(fpr)

                        roc_auc = auc(fpr, tpr)
                        roc_aucs[ood_image_filename] = roc_auc

                mean_iou = sum(ious.values()) / len(ious)
                mean_auc = sum(roc_aucs.values()) / len(roc_aucs)
                logging.info(f'mean IoU: {mean_iou}')
                logging.info(f'mean AUC: {mean_auc}')

                cams_save_path = os.path.join(activation_view_path, 'cams.pkl')
                ious_save_path = os.path.join(activation_view_path, 'ious.pkl')
                roc_aucs_save_path = os.path.join(activation_view_path, 'roc_aucs.pkl')
                with open(cams_save_path, 'wb') as f:
                    pickle.dump(cams, f)

                with open(ious_save_path, 'wb') as f:
                    pickle.dump(ious, f)

                with open(roc_aucs_save_path, 'wb') as f:
                    pickle.dump(roc_aucs, f)


if __name__ == '__main__':
    main()
