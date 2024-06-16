import argparse
import glob
import os
import pickle

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

from analysis.zernike_moments import compute_zernike_moments
from util.util import find_label_by_identifier


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse the shapes of object using Zernike moments')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--data-dir', type=str, help='Root path to the datasets')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save plots')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    data_dir = args.data_dir
    log_dir = args.log_dir
    plot_dir = args.plot_dir
    zernike_moments = {}
    zernike_moments_by_category = {}

    for data_folder in config['data']['data_folders']:
        print(f"Processing data folder {data_folder}")
        zernike_moments[data_folder] = {}
        zernike_moments_by_category[data_folder] = {}

        # compute the zernike moments of each image
        image_paths = glob.glob(os.path.join(data_dir, data_folder, '*.png'))
        for image_path in tqdm(image_paths, desc='Processing images'):
            image_name = image_path.split('/')[-1]
            zernike_moments[data_folder][image_name] = compute_zernike_moments(image_path)

            # sort the moments by category
            image_identifier = image_name.split('_')[0]
            category = find_label_by_identifier(image_identifier)
            if category not in zernike_moments_by_category[data_folder]:
                zernike_moments_by_category[data_folder][category] = []
            zernike_moments_by_category[data_folder][category].append(zernike_moments[data_folder][image_name])
            sorted_zernike_moments_by_category = {k: zernike_moments_by_category[data_folder][k]
                                                  for k in sorted(zernike_moments_by_category[data_folder].keys())}
            zernike_moments_by_category[data_folder] = sorted_zernike_moments_by_category

        # save the results
        log_path = os.path.join(log_dir, data_folder)
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_dir, data_folder, 'zernike_moments.pkl'), 'wb') as f:
            pickle.dump(zernike_moments, f)
        with open(os.path.join(log_dir, data_folder, 'zernike_moments_by_category.pkl'), 'wb') as f:
            pickle.dump(zernike_moments_by_category, f)

        plot_path = os.path.join(plot_dir, data_folder)
        os.makedirs(plot_path, exist_ok=True)


def plot_zernike_moments_by_category():
    args = parse_args()
    category_labels = {  # Map numerical labels to category names
        0: 'airplane', 1: 'backpack', 2: 'basket', 3: 'bed', 4: 'bicycle', 5: 'bread',
        6: 'cabinet', 7: 'cake', 8: 'camera', 9: 'candle', 10: 'car_(automobile)', 11: 'chair',
        12: 'clock', 13: 'cone', 14: 'frying_pan', 15: 'hat', 16: 'jacket', 17: 'laptop_computer',
        18: 'microwave_oven', 19: 'motorcycle', 20: 'pie', 21: 'pizza', 22: 'sandwich', 23: 'shirt',
        24: 'shoe', 25: 'sofa', 26: 'street_sign', 27: 'sweat_pants', 28: 'table', 29: 'television_set',
        30: 'trash_can', 31: 'truck'
    }

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    log_dir = args.log_dir
    plot_dir = args.plot_dir

    # Dictionary to store RMS values for each category
    rms_values_by_category = {}

    for data_folder in config['data']['data_folders']:
        with open(os.path.join(log_dir, data_folder, 'zernike_moments_by_category.pkl'), 'rb') as f:
            zernike_moments_by_category = pickle.load(f)

        for category, moments in zernike_moments_by_category[data_folder].items():
            moments_array = np.array(moments)

            # Compute the RMS of Zernike moments for each sample
            rms_values = np.sqrt(np.mean(np.square(moments_array), axis=1))

            # Store the RMS values for the category
            if category not in rms_values_by_category:
                rms_values_by_category[category] = []
            rms_values_by_category[category].extend(rms_values)

            # Plotting mean and std of Zernike moments for the current category
            mean_moments = np.mean(moments_array, axis=0)
            std_moments = np.std(moments_array, axis=0)

            plt.figure()
            x = np.arange(len(mean_moments))
            plt.errorbar(x, mean_moments, yerr=std_moments, fmt='-o', capsize=5)
            plt.xlabel('Zernike Moment Index')
            plt.ylabel('Value')
            plt.title(f'Mean and Std of Zernike Moments for {category} in {data_folder}')

            # Save the plot
            plot_filename = os.path.join(plot_dir, data_folder, f'{category}_zernike_moments.png')
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            plt.savefig(plot_filename)
            plt.close()

        # Combine RMS values into one plot for all categories
        plt.figure(figsize=(18, 12))  # Set figure size to be larger
        for category, rms_values in rms_values_by_category.items():
            rms_values = np.array(rms_values)
            mean_rms = np.mean(rms_values)
            std_rms = np.std(rms_values)

            plt.errorbar([category], [mean_rms], yerr=[std_rms], fmt='o', capsize=5, label=f'{category}')

        plt.xlabel('Category')
        plt.ylabel('RMS Value')
        plt.title('RMS of Zernike Moments by Category')
        plt.xticks(rotation=45)  # Tilt x-axis labels for better visibility

        # Save the combined RMS plot
        rms_plot_filename = os.path.join(plot_dir, data_folder, 'rms_zernike_moments_by_category.png')
        os.makedirs(os.path.dirname(rms_plot_filename), exist_ok=True)
        plt.savefig(rms_plot_filename)
        plt.close()

        # correlation between shape bias and Zernike moments
        shape_bias_by_category = {label: [] for label in category_labels.values()}
        texture_bias_by_category = {label: [] for label in category_labels.values()}

        for log_folder in config['log']['log_folders']:
            result_path = os.path.join(log_dir, log_folder, config['log']['cue_conflict'], 'results.pkl')
            with open(result_path, 'rb') as file:
                results = pickle.load(file)

            shape_decisions = results['shape_decisions'][0]
            texture_decisions = results['texture_decisions'][0]

            for category_num, shape_decision in shape_decisions.items():
                total_decisions = shape_decisions[category_num] + texture_decisions[category_num]
                category_name = category_labels[category_num]
                if total_decisions > 0:
                    shape_bias_by_category[category_name].append(shape_decisions[category_num] / total_decisions)
                else:
                    shape_bias_by_category[category_name].append(0)

        mean_shape_bias_by_category = {label: np.mean(biases) for label, biases in shape_bias_by_category.items()}
        #mean_texture_bias_by_category = {label: np.mean(biases) for label, biases in texture_bias_by_category.items()}

        mean_rms_by_category = {category: np.mean(rms) for category, rms in rms_values_by_category.items()}
        std_rms_by_category = {category: np.std(rms) for category, rms in rms_values_by_category.items()}
        # Align categories and compute correlation
        categories = sorted(mean_shape_bias_by_category.keys())
        shape_biases = [mean_shape_bias_by_category[category] for category in categories if
                        category in mean_rms_by_category]
        rms_values = [mean_rms_by_category[category] for category in categories if category in mean_rms_by_category]
        rms_stds = [std_rms_by_category[category] for category in categories if category in std_rms_by_category]

        # Compute correlation
        correlation, p_value = pearsonr(rms_values, shape_biases)
        print(f"Pearson correlation coefficient (mean vs. bias): {correlation}, p-value: {p_value}")

        # Plot the correlation
        plt.figure(figsize=(10, 6))
        plt.scatter(rms_values, shape_biases, color='blue', label=f'Correlation: {correlation:.2f}')
        for i, category in enumerate(categories):
            if category in mean_rms_by_category:
                plt.text(rms_values[i], shape_biases[i], category, fontsize=9)

        plt.xlabel('RMS of Zernike Moments')
        plt.ylabel('Mean Shape Bias')
        plt.title('Correlation between RMS of Zernike Moments and Shape Bias by Category')
        plt.tight_layout()

        # Save the correlation plot
        correlation_plot_filename = os.path.join(plot_dir, data_folder, 'zernike_rms_shape_bias_correlation.png')
        os.makedirs(os.path.dirname(correlation_plot_filename), exist_ok=True)
        plt.savefig(correlation_plot_filename)
        plt.show()

        correlation, p_value = pearsonr(rms_stds, shape_biases)
        print(f"Pearson correlation coefficient (std vs. bias): {correlation}, p-value: {p_value}")

        # Plot the correlation std
        plt.figure(figsize=(10, 6))
        plt.scatter(rms_stds, shape_biases, color='red', label=f'Correlation: {correlation:.2f}')
        for i, category in enumerate(categories):
            if category in std_rms_by_category:
                plt.text(rms_stds[i], shape_biases[i], category, fontsize=9)

        plt.xlabel('Standard Deviation of RMS of Zernike Moments')
        plt.ylabel('Mean Shape Bias')
        plt.title('Correlation between Std of RMS of Zernike Moments and Shape Bias by Category')
        plt.tight_layout()

        # Save the correlation plot
        correlation_plot_filename = os.path.join(plot_dir, data_folder, 'zernike_rms_std_shape_bias_correlation.png')
        os.makedirs(os.path.dirname(correlation_plot_filename), exist_ok=True)
        plt.savefig(correlation_plot_filename)
        plt.show()


if __name__ == '__main__':
    #main()
    plot_zernike_moments_by_category()
    print('Done')
