import argparse
import os
import pickle
import sys
import warnings

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.models import get_device

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Texture bias evaluation on Cue-Conflict Dataset')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def plot_iou(ious, save_path):
    """
        Plots a histogram of IoU values, displays the mean and standard deviation,
        and saves the plot to the specified path.

        Parameters:
        - ious: Dictionary mapping image filenames to their IoU values.
        - save_path: The file path where the plot image will be saved.
        """
    # Extract IoU values
    iou_values = list(ious.values())
    iou_array = np.array(iou_values)

    # Calculate mean and standard deviation
    mean_iou = np.mean(iou_array)
    std_iou = np.std(iou_array)

    # Create histogram
    plt.figure(figsize=(8, 6))
    plt.hist(iou_array, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Add mean and std lines
    plt.axvline(mean_iou, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_iou:.2f}')
    plt.axvline(mean_iou - std_iou, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_iou:.2f}')
    plt.axvline(mean_iou + std_iou, color='green', linestyle='dashed', linewidth=1)

    # Add labels and title
    plt.xlabel('IoU Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 6000)
    plt.title('Distribution of IoU Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f'IoU histogram saved to {save_path}')


def plot_auc(roc_aucs, save_path):
    """
    Plots a histogram of ROC AUC values, displays the mean and standard deviation,
    and saves the plot to the specified path.

    Parameters:
    - roc_aucs: Dictionary mapping image filenames to their ROC AUC values.
    - save_path: The file path where the plot image will be saved.
    """
    # Extract ROC AUC values
    auc_values = list(roc_aucs.values())
    auc_array = np.array(auc_values)

    # Calculate mean and standard deviation
    mean_auc = np.mean(auc_array)
    std_auc = np.std(auc_array)

    # Create histogram
    plt.figure(figsize=(8, 6))
    plt.hist(auc_array, bins=20, color='salmon', edgecolor='black', alpha=0.7)

    # Add mean and std lines
    plt.axvline(mean_auc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_auc:.2f}')
    plt.axvline(mean_auc - std_auc, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_auc:.2f}')
    plt.axvline(mean_auc + std_auc, color='green', linestyle='dashed', linewidth=1)

    # Add labels and title
    plt.xlabel('ROC AUC Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 22500)
    plt.title('Distribution of ROC AUC Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f'ROC AUC histogram saved to {save_path}')


def plot_average_cam(cams, save_path):
    """
    Computes the average CAM across all images and plots it.

    Parameters:
    - cams: Dictionary mapping image filenames to their CAM arrays.
    - save_path: The file path where the average CAM image will be saved.
    """
    # Collect all CAM arrays into a list
    cam_arrays = list(cams.values())

    # Ensure all CAMs have the same shape
    cam_shapes = [cam.shape for cam in cam_arrays]
    if len(set(cam_shapes)) > 1:
        raise ValueError("All CAMs must have the same shape to compute the average.")

    # Stack CAM arrays and compute the mean across the first axis (images)
    cam_stack = np.stack(cam_arrays, axis=0)
    average_cam = np.mean(cam_stack, axis=0)

    # Normalize the average CAM to [0, 1] for visualization
    # average_cam_normalized = (average_cam - np.min(average_cam)) / (np.max(average_cam) - np.min(average_cam))
    # average_cam_normalized = np.squeeze(average_cam_normalized)
    average_cam = np.squeeze(average_cam)

    # Plot the average CAM
    plt.figure(figsize=(6, 6))
    plt.imshow(average_cam, cmap='jet')
    plt.colorbar(label='Activation')
    plt.title('Average CAM')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Average CAM image saved to {save_path}')


def plot_average_roc_curve(cams, ground_truth_masks, save_path):
    """
    Computes and plots the average ROC curve across all images using the CAMs.

    Parameters:
    - cams: Dictionary mapping image filenames to their CAM arrays.
    - ground_truth_masks: Dictionary mapping image filenames to their ground truth masks (binary arrays).
    - save_path: The file path where the ROC curve image will be saved.
    """
    thresholds = np.linspace(0, 1, 101)
    tprs = []
    fprs = []

    # Loop over all images
    for filename in tqdm(cams.keys()):
        cam = cams[filename]
        gt_mask = ground_truth_masks[filename]

        # Flatten the CAM and ground truth mask
        cam_flat = cam.flatten()
        gt_flat = gt_mask.flatten()

        # Normalize the CAM
        cam_normalized = (cam_flat - np.min(cam_flat)) / (np.max(cam_flat) - np.min(cam_flat))

        # Compute TPR and FPR at all thresholds
        fpr, tpr, _ = roc_curve(gt_flat, cam_normalized)
        tprs.append(np.interp(thresholds, fpr[::-1], tpr[::-1]))  # Interpolate TPR
        fprs.append(np.interp(thresholds, fpr[::-1], fpr[::-1]))  # Interpolate FPR

    # Compute the mean TPR and FPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)

    # Compute the AUC
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot the average ROC curve
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=f'Average ROC curve (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Across Images')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Average ROC curve saved to {save_path}')


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    log_dir = args.log_dir
    plot_dir = args.plot_dir

    model_folders = config['model']['model_folders']
    model_name = config['model']['model_name']
    epochs = [30]

    device = get_device()

    root = config['data']['root']
    views = config['data']['views']
    backgrounds = config['data']['backgrounds']
    resolution = config['data']['resolution']

    for model_folder in model_folders:
        for epoch in epochs:
            for view in views:
                metric_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view)

                ious_path = os.path.join(metric_path, 'ious.pkl')
                ious = pickle.load(open(ious_path, 'rb'))
                roc_aucs_path = os.path.join(metric_path, 'roc_aucs.pkl')
                roc_aucs = pickle.load(open(roc_aucs_path, 'rb'))
                # cams_path = os.path.join(metric_path, 'cams.pkl')
                # cams = pickle.load(open(cams_path, 'rb'))

                save_folder = os.path.join(plot_dir, 'activations', model_folder, str(epoch), view)
                os.makedirs(save_folder, exist_ok=True)

                ious_plot_save_path = os.path.join(save_folder, f'ious_{model_folder}_{epoch}_on_{view}.png')
                plot_iou(ious, ious_plot_save_path)

                roc_aucs_plot_save_path = os.path.join(save_folder, f'auc_{model_folder}_{epoch}_on_{view}.png')
                plot_auc(roc_aucs, roc_aucs_plot_save_path)

                # Temp: reconstruction the mean roc curve
                # gt_folders = []
                # for background in backgrounds:
                #     gt_folders.append(os.path.join(root, f'gt_ood_{background}_{view}_{resolution}'))
                # image_filenames = list(cams.keys())
                # ground_truth_masks = {}
                # for image_filename in tqdm(image_filenames, desc='Fetching the ground truth masks'):
                #     for gt_folder in gt_folders:
                #         gt_image_path = os.path.join(gt_folder, image_filename)
                #         if os.path.exists(gt_image_path):
                #             gt_image = Image.open(gt_image_path)
                #             gt_alpha_channel = gt_image.split()[-1].convert('L')
                #             gt_mask = np.array(gt_alpha_channel)
                #             gt_mask = cv2.resize(gt_mask, (224, 224))
                #             gt_mask = (gt_mask > 0).astype(np.uint8)
                #             ground_truth_masks[image_filename] = gt_mask
                #         else:
                #             continue

                # avg_cam_save_path = os.path.join(save_folder, f'avg_cam_{model_folder}_{epoch}_on_{view}.png')
                # plot_average_cam(cams, avg_cam_save_path)

                # avg_roc_save_path = os.path.join(save_folder, f'avg_roc_{model_folder}_{epoch}_on_{view}.png')
                # plot_average_roc_curve(cams, ground_truth_masks, avg_roc_save_path)


if __name__ == '__main__':
    main()
