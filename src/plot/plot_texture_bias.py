import argparse
import os
import pickle

import yaml
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
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
    acc_min = config['plot']['acc_min']
    acc_max = config['plot']['acc_max']

    for log_folder in log_folders:
        result_path = os.path.join(log_dir, log_folder, cue_conflict_folder, 'results.pkl')
        with open(result_path, 'rb') as file:
            results = pickle.load(file)

        content_acc = results['content_accuracies']
        epochs = range(0, len(results['content_accuracies']) + 1)

        shape_bias = []
        cue_conflict_acc = []
        for i in range(len(epochs) - 1):
            shape_bias.append(results['shape_decisions'][i] / (
                    results['texture_decisions'][i] + results['shape_decisions'][i]))
            cue_conflict_acc.append(
                results['shape_decisions'][i] + results['texture_decisions'][i])

        shape_bias.insert(0, 0.5)
        cue_conflict_acc.insert(0, 0.0625)
        content_acc.insert(0, 0.03125)

        plt.plot(epochs, shape_bias, label='Shape Bias')

        plt.plot(epochs, content_acc, label='Content Accuracy')

        plt.plot(epochs, cue_conflict_acc, label='Cue Conflict Accuracy')

        plt.xlabel('Epoch')
        plt.xticks(range(0, len(epochs), 5))
        plt.ylabel('Accuracy')
        plt.ylim([acc_min, acc_max])
        title = '_'.join([model_name, log_folder, cue_conflict_folder])
        plt.title(title)
        plt.legend(loc='upper right')

        os.makedirs(os.path.join(plot_dir, log_folder), exist_ok=True)
        save_path = os.path.join(plot_dir, log_folder, '_'.join([title, 'cue_conflict.png']))
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    main()
    print('Done')
