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
                        interior_cam_values = cam[interior_mask].flatten()
                        exterior_cam_values = cam[exterior_mask].flatten()
                        border_cam_values = cam[border_mask].flatten()

                        # (3) save the CAM values
                        all_interior_cam_values.append(interior_cam_values)
                        all_exterior_cam_values.append(exterior_cam_values)
                        all_border_cam_values.append(border_cam_values)
                    else:
                        logging.warning(f'Analysis aborted for {image_filename}')

                # all_cam_values = {
                #     'interior': all_interior_cam_values,
                #     'border': all_border_cam_values,
                #     'exterior': all_exterior_cam_values,
                # }
                del cams

                # (1) Calculate the mean CAM values for each image and store in new lists
                interior_means_per_image = [np.mean(cam_values) for cam_values in all_interior_cam_values]
                border_means_per_image = [np.mean(cam_values) for cam_values in all_border_cam_values]
                exterior_means_per_image = [np.mean(cam_values) for cam_values in all_exterior_cam_values]

                interior_pixels_per_image = [len(cam_values) for cam_values in all_interior_cam_values]
                border_pixels_per_image = [len(cam_values) for cam_values in all_border_cam_values]
                exterior_pixels_per_image = [len(cam_values) for cam_values in all_exterior_cam_values]

                # (2) Calculate the global mean of each list
                global_interior_mean = np.mean(interior_pixels_per_image)
                global_border_mean = np.mean(border_pixels_per_image)
                global_exterior_mean = np.mean(exterior_pixels_per_image)

                mean_cams = {
                    'image_means': {
                        'interior_means_per_image': interior_means_per_image,
                        'border_means_per_image': border_means_per_image,
                        'exterior_means_per_image': exterior_means_per_image
                    },
                    'pixels': {
                        'interior_pixels_per_image': interior_pixels_per_image,
                        'border_pixels_per_image': border_pixels_per_image,
                        'exterior_pixels_per_image': exterior_pixels_per_image
                    },
                    'global_means': {
                        'global_interior_mean': global_interior_mean,
                        'global_border_mean': global_border_mean,
                        'global_exterior_mean': global_exterior_mean
                    },
                }

                mean_cams_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'mean_cams.pkl')
                os.makedirs(os.path.dirname(mean_cams_save_path), exist_ok=True)
                with open(mean_cams_save_path, 'wb') as f:
                    pickle.dump(mean_cams, f)
                del interior_means_per_image, border_means_per_image, exterior_means_per_image
                del interior_pixels_per_image, border_pixels_per_image, exterior_pixels_per_image
                # del all_interior_cam_values, all_border_cam_values, all_exterior_cam_values
                logging.info(f'Saved mean_cams')

                # Flatten the lists of CAM values for global comparison
                flattened_interior_cams = [item for sublist in all_interior_cam_values for item in sublist]
                del all_interior_cam_values
                flattened_border_cams = [item for sublist in all_border_cam_values for item in sublist]
                del all_border_cam_values
                flattened_exterior_cams = [item for sublist in all_exterior_cam_values for item in sublist]
                del all_exterior_cam_values

                # (3) Compare distributions using t-tests between interior, border, and exterior regions
                t_test_interior_vs_exterior = stats.ttest_ind(flattened_interior_cams, flattened_exterior_cams)
                t_test_interior_vs_border = stats.ttest_ind(flattened_interior_cams, flattened_border_cams)
                t_test_exterior_vs_border = stats.ttest_ind(flattened_exterior_cams, flattened_border_cams)

                # (4) Save the statistics, including the means and test results
                statistics = {
                    't_tests': {
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
                }

                # Save the statistics as a pickle file
                stats_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch),
                                               'partitioned_cams_stats.pkl')
                os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
                with open(stats_save_path, 'wb') as stats_file:
                    pickle.dump(statistics, stats_file)
                logging.info(f'Saved statistics')

                logging.info(f"Model: {model_folder}, Epoch: {epoch}")
                logging.info(f"Global Interior Mean: {global_interior_mean}")
                logging.info(f"Global Exterior Mean: {global_exterior_mean}")
                logging.info(f"Global Border Mean: {global_border_mean}")
                logging.info(f"Interior vs Exterior T-test: {t_test_interior_vs_exterior}")
                logging.info(f"Interior vs Border T-test: {t_test_interior_vs_border}")
                logging.info(f"Exterior vs Border T-test: {t_test_exterior_vs_border}")
                logging.info("-------------------------------------------------------")

                # interior_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'interior_cams.pkl')
                # with open(interior_save_path, 'wb') as f:
                #     pickle.dump(all_interior_cam_values, f)
                # del all_interior_cam_values
                # logging.info(f'Interior CAM saved at {interior_save_path}')
                # border_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'border_cams.pkl')
                # with open(border_save_path, 'wb') as f:
                #     pickle.dump(all_border_cam_values, f)
                # del all_border_cam_values
                # logging.info(f'Border CAM saved at {border_save_path}')
                # exterior_save_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, 'exterior_cams.pkl')
                # with open(exterior_save_path, 'wb') as f:
                #     pickle.dump(all_exterior_cam_values, f)
                # del all_exterior_cam_values
                # logging.info(f'Exterior CAM saved at {exterior_save_path}')


if __name__ == '__main__':
    main()
