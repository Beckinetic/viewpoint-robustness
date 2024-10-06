import argparse
import json
import logging
import os
import pickle
import sys

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFilter
from scipy import stats
from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run picture distortion')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, help='Path to the logs directory')
    return parser.parse_args()


def find_gt_image(distorted_image_filename, gt_folders):
    for gt_folder in gt_folders:
        gt_image_path = os.path.join(gt_folder, distorted_image_filename)
        if os.path.exists(gt_image_path):
            return gt_image_path

    logging.warning(f'Could not find ground truth image for {distorted_image_filename}')
    return None


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    log_dir = args.log_dir

    model_folders = config['model']['model_folders']
    epochs = [30]

    root = config['data']['root']
    views = config['data']['views']

    for model_folder in model_folders:
        for epoch in epochs:
            for view in views:
                logging.info(f'Processing Model {model_folder}, Epoch {epoch}, View {view}')
                gt_folders = [os.path.join(root, f'gt_ood_meadow_{view}_f'),
                              os.path.join(root, f'gt_ood_forest_{view}_f'),
                              os.path.join(root, f'gt_ood_desert_{view}_f'),
                              os.path.join(root, f'gt_ood_industrial_{view}_f'),]
                cams_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'cams.pkl')
                with open(cams_path, 'rb') as f:
                    cams = pickle.load(f)

                all_interior_cam_values = []
                all_border_cam_values = []
                all_exterior_cam_values = []

                for image_filename, cam in tqdm(cams.items()):
                    gt_image_path = find_gt_image(image_filename, gt_folders)
                    if gt_image_path is not None:
                        gt_image = Image.open(gt_image_path)

                        # Extract alpha channel (object mask)
                        alpha_channel = gt_image.split()[-1].convert('L')
                        alpha_channel = alpha_channel.resize((224, 224), Image.LANCZOS)

                        # (1) Detect object border, interior, and exterior
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

                        # (2) extract CAM values for each region
                        cam = np.squeeze(np.array(cam))
                        # logging.info(f'Cam shape {cam.shape}')
                        # logging.info(f'Interior shape {interior_mask.shape}')
                        # logging.info(f'Exterior shape {exterior_mask.shape}')
                        interior_cam_values = np.multiply(cam, interior_mask).flatten()
                        exterior_cam_values = np.multiply(cam, exterior_mask).flatten()
                        border_cam_values = np.multiply(cam, border_mask).flatten()

                        # (3) save the CAM values
                        all_interior_cam_values.extend(interior_cam_values)
                        all_exterior_cam_values.extend(exterior_cam_values)
                        all_border_cam_values.extend(border_cam_values)
                    else:
                        logging.warning(f'Analysis aborted for {image_filename}')

                # all_cam_values = {
                #     'interior': all_interior_cam_values,
                #     'border': all_border_cam_values,
                #     'exterior': all_exterior_cam_values,
                # }
                del cams

                # (3) Run statistical analysis on the collected CAM values
                interior_mean = np.mean(all_interior_cam_values)
                border_mean = np.mean(all_border_cam_values)
                exterior_mean = np.mean(all_exterior_cam_values)

                # Compare distributions using t-test
                t_test_interior_vs_exterior = stats.ttest_ind(all_interior_cam_values, all_exterior_cam_values)
                t_test_interior_vs_border = stats.ttest_ind(all_interior_cam_values, all_border_cam_values)
                t_test_exterior_vs_border = stats.ttest_ind(all_exterior_cam_values, all_border_cam_values)

                statistics = {
                    'interior_mean': interior_mean,
                    'border_mean': border_mean,
                    'exterior_mean': exterior_mean,
                    't_test_interior_vs_exterior': {
                        'statistic': t_test_interior_vs_exterior.statistic,
                        'pvalue': t_test_interior_vs_exterior.pvalue
                    },
                    't_test_interior_vs_border': {
                        'statistic': t_test_interior_vs_border.statistic,
                        'pvalue': t_test_interior_vs_border.pvalue
                    },
                    't_test_exterior_vs_border': {
                        'statistic': t_test_exterior_vs_border.statistic,
                        'pvalue': t_test_exterior_vs_border.pvalue
                    }
                }

                stats_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'partitioned_cams_stats.json')
                with open(stats_save_path, 'w') as stats_file:
                    json.dump(statistics, stats_file, indent=4)

                logging.info(f"Model: {model_folder}, Epoch: {epoch}, View: {view}")
                logging.info(f"Interior vs Exterior: {t_test_interior_vs_exterior}")
                logging.info(f"Interior vs Border: {t_test_interior_vs_border}")
                logging.info(f"Exterior vs Border: {t_test_exterior_vs_border}")
                logging.info("-------------------------------------------------------")

                interior_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'interior_cams.pkl')
                with open(interior_save_path, 'wb') as f:
                    pickle.dump(all_interior_cam_values, f)
                del all_interior_cam_values
                logging.info(f'Interior CAM saved at {interior_save_path}')
                border_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'border_cams.pkl')
                with open(border_save_path, 'wb') as f:
                    pickle.dump(all_border_cam_values, f)
                del all_border_cam_values
                logging.info(f'Border CAM saved at {border_save_path}')
                exterior_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'exterior_cams.pkl')
                with open(exterior_save_path, 'wb') as f:
                    pickle.dump(all_exterior_cam_values, f)
                del all_exterior_cam_values
                logging.info(f'Exterior CAM saved at {exterior_save_path}')


if __name__ == '__main__':
    main()
