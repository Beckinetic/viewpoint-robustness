import argparse
import logging
import os
import pickle
import sys
from itertools import combinations
from statsmodels.stats.multitest import multipletests

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy import stats

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run picture distortion')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, help='Path to the logs directory')
    parser.add_argument('--plot-dir', type=str, help='Path to the plot directory')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    log_dir = args.log_dir
    plot_dir = args.plot_dir

    model_folders = config['model']['model_folders']
    epochs = [30]

    root = config['data']['root']
    views = config['data']['views']

    cam_filenames = ['interior_cams.pkl', 'border_cams.pkl', 'exterior_cams.pkl']

    # compare model performances over the same dataset
    for view in views:
        for cam_filename in cam_filenames:
            cam_type = cam_filename.split('_')[0]
            logging.info(f'Comparing data from view {view} and cam type {cam_type}')
            for epoch in epochs:
                cams = []
                cam_labels = []
                for model_folder in model_folders:
                    logging.info(f'Loading data from {model_folder}, epoch {epoch}')
                    data_path = os.path.join(log_dir, model_folder, 'activations', str(epoch), view, cam_filename)
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    data = np.concatenate(data)
                    cams.append(data)
                    cam_labels.append(model_folder)

                # statistics
                # one-way ANOVA
                anova_results = {}
                f_statistic, p_value = stats.f_oneway(*cams)
                anova_results['F-statistic'] = f_statistic
                anova_results['p-value'] = p_value
                logging.info(f"ANOVA Results for view {view}, cam type {cam_type}, epoch {epoch}:")
                logging.info(f"F-statistic: {f_statistic}, p-value: {p_value}")
                anova_save_path = os.path.join(log_dir, 'border_cam_values', str(epoch), view, cam_type,
                                               'anova_results.pkl')
                os.makedirs(os.path.dirname(anova_save_path), exist_ok=True)
                with open(anova_save_path, 'wb') as f:
                    pickle.dump(anova_results, f)

                # pairwise comparisons using t-test with Bonferroni correction
                pairwise_comparisons = list(combinations(range(len(cams)), 2))  # Generate all pairs of models
                p_values = []
                comparisons = []

                for i, j in pairwise_comparisons:
                    cam1 = cams[i]
                    cam2 = cams[j]
                    t_stat, p_val = stats.ttest_ind(cam1, cam2)  # Perform t-test between each pair of models
                    p_values.append(p_val)
                    comparisons.append((cam_labels[i], cam_labels[j]))

                # apply Bonferroni correction
                corrected_p_values = multipletests(p_values, alpha=0.05, method='bonferroni')[1]

                logging.info("Bonferroni corrected p-values for pairwise comparisons:")
                for (model1, model2), p_val, corrected_p_val in zip(comparisons, p_values, corrected_p_values):
                    logging.info(
                        f"{model1} vs {model2}: raw p-value = {p_val:.3e}, corrected p-value = {corrected_p_val:.3e}")

                pairwise_comparisons = {
                    'comparisons': comparisons,
                    'p_values': p_values,
                    'corrected_p_values': corrected_p_values,
                    'method': 'bonferroni'
                }

                pairwise_save_path = os.path.join(log_dir, 'border_cam_values', str(epoch), view,
                                                  cam_type, 'pairwise_results.pkl')
                os.makedirs(os.path.dirname(pairwise_save_path), exist_ok=True)
                with open(pairwise_save_path, 'wb') as f:
                    pickle.dump(pairwise_comparisons, f)

                # plot CAM value distributions
                plt.figure(figsize=(10, 6))

                # Plot KDE for each model's CAM values
                plt.figure(figsize=(10, 6))

                # Plot histograms for each model
                for i, cam in enumerate(cams):
                    mean = np.mean(cam)
                    std = np.std(cam)
                    cam_model = cam_labels[i]
                    plt.hist(cam, bins=100, alpha=0.5, density=True,
                             label=f'{cam_model} (mean: {mean:.2f}, std: {std:.2f})')

                plt.title(f"CAM Value Distributions - View: {view}, Cam Type: {cam_type}, Epoch: {epoch}")
                plt.xlabel('CAM Values')
                plt.ylabel('Density')
                plt.ylim(0, 3.5)
                plt.legend(loc='best')

                # display significance results on the plot
                # significant_pairs = [corrected_p_val < 0.05 for corrected_p_val in corrected_p_values]

                # annotate the plot with significant pairwise comparisons after Bonferroni correction
                # for i, (model1, model2) in enumerate(comparisons):
                #     if significant_pairs[i]:
                #         plt.text(0.05, 0.95 - i * 0.05,
                #                  f"Significant: {model1} vs {model2} (p={corrected_p_values[i]:.3e})",
                #                  transform=plt.gca().transAxes)

                plt.tight_layout()
                plot_path = os.path.join(plot_dir, 'border_cam_values', f'cam_value_dist_{view}_{epoch}_{cam_type}.png')
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)


if __name__ == '__main__':
    main()
