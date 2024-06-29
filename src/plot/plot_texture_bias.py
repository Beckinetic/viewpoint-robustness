import argparse
import os
import pickle
import yaml
from matplotlib import pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save plots')
    return parser.parse_args()


def main():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)
    log_dir = args.log_dir
    plot_dir = args.plot_dir

    log_folders = config['log']['log_folders']
    cue_conflict_folder = config['log']['cue_conflict']
    model_name = config['model']['model_name']
    prefix = config['plot']['prefix']
    acc_min = config['plot']['acc_min']
    acc_max = config['plot']['acc_max']

    category_labels = {  # Map numerical labels to category names
        0: 'airplane', 1: 'backpack', 2: 'basket', 3: 'bed', 4: 'bicycle', 5: 'bread',
        6: 'cabinet', 7: 'cake', 8: 'camera', 9: 'candle', 10: 'car_(automobile)', 11: 'chair',
        12: 'clock', 13: 'cone', 14: 'frying_pan', 15: 'hat', 16: 'jacket', 17: 'laptop_computer',
        18: 'microwave_oven', 19: 'motorcycle', 20: 'pie', 21: 'pizza', 22: 'sandwich', 23: 'shirt',
        24: 'shoe', 25: 'sofa', 26: 'street_sign', 27: 'sweat_pants', 28: 'table', 29: 'television_set',
        30: 'trash_can', 31: 'truck'
    }

    model_names = []
    content_accuracies = []
    shape_biases = []
    texture_biases = []
    total_shape_decisions = []
    total_texture_decisions = []

    shape_bias_by_category = {label: [] for label in category_labels.keys()}
    texture_bias_by_category = {label: [] for label in category_labels.keys()}

    for log_folder in log_folders:
        if config['log']['pasted']:
            result_file_name = 'shape_bias_pasted.pkl'
        else:
            result_file_name = 'shape_bias.pkl'
        result_path = os.path.join(log_dir, log_folder, cue_conflict_folder, result_file_name)
        with open(result_path, 'rb') as file:
            results = pickle.load(file)

        # Extract metrics
        content_accuracy = results['content_accuracies'][0]
        shape_decisions = results['shape_decisions'][0]
        texture_decisions = results['texture_decisions'][0]
        all_decisions = results['total_decisions'][0]

        total_shape_decision_count = sum(shape_decisions.values())
        total_texture_decision_count = sum(texture_decisions.values())
        all_decisions_count = sum(all_decisions.values())
        overall_decision_count = total_shape_decision_count + total_texture_decision_count

        shape_bias = np.sqrt(total_shape_decision_count / overall_decision_count) * np.sqrt(total_shape_decision_count / all_decisions_count)
        texture_bias = np.sqrt(total_texture_decision_count / overall_decision_count) * np.sqrt(total_texture_decision_count / all_decisions_count)

        label_mapping = {'combined_f_combined': 'combined, free view',
                         'combined_r_combined': 'combined, restricted view',
                         'combined_fx_combined': 'combined, fixed view',
                         'meadow_f_f': 'meadow, free view',
                         'meadow_r_f': 'meadow, restricted view',
                         'meadow_fx_f': 'meadow, fixed view',
                         'forest_f_f': 'forest, free view',
                         'forest_r_f': 'forest, restricted view',
                         'forest_fx_f': 'forest, fixed view',
                         'desert_f_f': 'desert, free view',
                         'desert_r_f': 'desert, restricted view',
                         'desert_fx_f': 'desert, fixed view',
                         'industrial_f_f': 'industrial area, free view',
                         'industrial_r_f': 'industrial, restricted view',
                         'industrial_fx_f': 'industrial, fixed view',
                         'mixed_f_f_0.1': 'mixed, free view, r = 0.1',
                         'mixed_r_f_0.1': 'mixed, restricted view, r = 0.1',
                         'mixed_fx_f_0.1': 'mixed, fixed view, r = 0.1',
                         'mixed_f_f_0.25': 'mixed, free view, r = 0.25',
                         'mixed_r_f_0.25': 'mixed, restricted view, r = 0.25',
                         'mixed_fx_f_0.25': 'mixed, fixed view, r = 0.25',
                         'mixed_f_f_0.3': 'mixed, free view, r = 0.3',
                         'mixed_r_f_0.3': 'mixed, restricted view, r = 0.3',
                         'mixed_fx_f_0.3': 'mixed, fixed view, r = 0.3',
                         'mixed_f_f_0.5': 'mixed, free view, r = 0.5',
                         'mixed_r_f_0.5': 'mixed, restricted view, r = 0.5',
                         'mixed_fx_f_0.5': 'mixed, fixed view, r = 0.5',
                         'mixed_f_f_0.7': 'mixed, free view, r = 0.7',
                         'mixed_r_f_0.7': 'mixed, restricted view, r = 0.7',
                         'mixed_fx_f_0.7': 'mixed, fixed view, r = 0.7',
                         'mixed_f_f_0.9': 'mixed, free view, r = 0.9',
                         'mixed_r_f_0.9': 'mixed, restricted view, r = 0.9',
                         'mixed_fx_f_0.9': 'mixed, fixed view, r = 0.9'}
        model_names.append(label_mapping[log_folder])  # Use log_folder name as model identifier
        content_accuracies.append(content_accuracy)
        shape_biases.append(shape_bias)
        texture_biases.append(texture_bias)
        total_shape_decisions.append(total_shape_decision_count)
        total_texture_decisions.append(total_texture_decision_count)

        for category in shape_decisions:
            total_decisions = shape_decisions[category] + texture_decisions[category]
            if total_decisions > 0:
                shape_bias_by_category[category].append(shape_decisions[category] / total_decisions)
                texture_bias_by_category[category].append(texture_decisions[category] / total_decisions)
            else:
                shape_bias_by_category[category].append(0)
                texture_bias_by_category[category].append(0)

    # Aggregate shape and texture biases by category across all models
    mean_shape_bias_by_category = {label: np.mean(biases) for label, biases in shape_bias_by_category.items()}
    mean_texture_bias_by_category = {label: np.mean(biases) for label, biases in texture_bias_by_category.items()}

    # print everything
    print(f"model names:{model_names}")
    print(f"content accuracies:{content_accuracies}")
    print(f"shape biases:{shape_biases}")
    print(f"texture biases:{texture_biases}")
    print(f"shape decisions:{total_shape_decisions}")
    print(f"texture decisions:{total_texture_decisions}")

    # Plotting content accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, content_accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Content Accuracy')
    plt.title('Content Accuracies of Different Models')
    plt.ylim(acc_min, acc_max)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'content_accuracies.png'])))
    plt.show()

    # Plotting overall shape bias
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, shape_biases, color='salmon')
    plt.xlabel('Model')
    plt.ylabel('Overall Shape Bias')
    plt.title('Overall Shape Bias of Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'overall_shape_bias.png'])))
    plt.show()

    # Plotting overall texture bias
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, texture_biases, color='lightgreen')
    plt.xlabel('Model')
    plt.ylabel('Overall Texture Bias')
    plt.title('Overall Texture Bias of Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'overall_texture_bias.png'])))
    plt.show()

    # Plotting overall count of shape or texture decisions
    plt.figure(figsize=(10, 6))
    width = 0.35
    indices = np.arange(len(model_names))
    plt.bar(indices - width/2, total_shape_decisions, width, label='Shape Decisions', color='steelblue')
    plt.bar(indices + width/2, total_texture_decisions, width, label='Texture Decisions', color='darkorange')
    plt.xlabel('Model')
    plt.ylabel('Decision Count')
    plt.title('Total Shape and Texture Decisions of Different Models')
    plt.xticks(indices, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'total_decisions.png'])))
    plt.show()

    # Plotting shape bias by category as scatter plot
    plt.figure(figsize=(15, 10))
    for model_idx, model in enumerate(model_names):
        biases = [shape_bias_by_category[category][model_idx] for category in category_labels.keys()]
        plt.scatter(category_labels.values(), biases, label=model)
    plt.xlabel('Category')
    plt.ylabel('Shape Bias by Category')
    plt.title('Shape Bias by Category for Different Models')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'shape_bias_by_category.png'])))
    plt.show()

    # Plotting texture bias by category as scatter plot
    plt.figure(figsize=(15, 10))
    for model_idx, model in enumerate(model_names):
        biases = [texture_bias_by_category[category][model_idx] for category in category_labels.keys()]
        plt.scatter(category_labels.values(), biases, label=model)
    plt.xlabel('Category')
    plt.ylabel('Texture Bias by Category')
    plt.title('Texture Bias by Category for Different Models')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'texture_bias_by_category.png'])))
    plt.show()

    # Plotting bar plots for mean shape bias by categories
    plt.figure(figsize=(15, 10))
    plt.bar(category_labels.values(), mean_shape_bias_by_category.values(), color='steelblue')
    plt.xlabel('Category')
    plt.ylabel('Mean Shape Bias')
    plt.title('Mean Shape Bias by Category Across All Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'mean_shape_bias_by_category.png'])))
    plt.show()

    # Plotting bar plots for mean texture bias by categories
    plt.figure(figsize=(15, 10))
    plt.bar(category_labels.values(), mean_texture_bias_by_category.values(), color='darkorange')
    plt.xlabel('Category')
    plt.ylabel('Mean Texture Bias')
    plt.title('Mean Texture Bias by Category Across All Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '_'.join([model_name, prefix, 'mean_texture_bias_by_category.png'])))
    plt.show()


if __name__ == '__main__':
    main()
