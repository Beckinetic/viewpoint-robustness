import argparse
import os
import pickle

import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib import patches


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def plot_mixed_loss_acc():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    biased_ratios = config['log']['biased_ratios']

    for view in config['log']['view']:
        for res in config['log']['res']:
            plot_folder = '_'.join(['mixed', view, res])
            # load data from all ratios
            min_training_losses = []
            min_validation_losses = []
            max_training_accuracies = []
            max_validation_accuracies = []
            for biased_ratio in biased_ratios:
                log_folder = '_'.join(['mixed', view, res, str(biased_ratio)])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))

                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                # access the variables
                min_training_losses.append(min(result['tl']))
                min_validation_losses.append([min(vl) for vl in result['vl']])
                max_training_accuracies.append(max(result['ta']))
                max_validation_accuracies.append([max(va) for va in result['va']])

            # making a heat map
            min_validation_losses_array = np.array(min_validation_losses)
            max_validation_accuracies_array = np.array(max_validation_accuracies)

            # create and save the heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(min_validation_losses_array, annot=True, cmap='coolwarm', vmin=config['plot']['loss_min'], vmax=config['plot']['loss_max'])
            title = '_'.join([model_name, "val_losses"])
            plt.title(title)
            plt.xticks(np.arange(len(biased_ratios)) + 0.5, biased_ratios, rotation=45)
            plt.yticks(np.arange(len(biased_ratios)) + 0.5, biased_ratios, rotation=45)
            rect = patches.Rectangle((1, 0), 1, len(min_validation_losses_array), linewidth=2, edgecolor='black',
                                     facecolor='none')
            ax.add_patch(rect)
            plt.xlabel('Validation Set Biased Ratio')
            plt.ylabel('Training Set Biased Ratio')

            os.makedirs(os.path.join(args.plot_dir, plot_folder), exist_ok=True)
            save_path = os.path.join(args.plot_dir, plot_folder, '_'.join([title, 'static.png']))
            plt.savefig(save_path)

            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(max_validation_accuracies_array, annot=True, cmap='coolwarm', vmin=config['plot']['acc_min'], vmax=config['plot']['acc_max'])
            title = '_'.join([model_name, "val_acc"])
            # plt.title(title)
            plt.xticks(np.arange(len(biased_ratios)) + 0.5, biased_ratios, rotation=45)
            plt.yticks(np.arange(len(biased_ratios)) + 0.5, biased_ratios, rotation=45)
            rect = patches.Rectangle((1, 0), 1, len(max_validation_accuracies_array), linewidth=2, edgecolor='black',
                                     facecolor='none')
            ax.add_patch(rect)
            plt.xlabel('Validation Set Biased Ratio')
            plt.ylabel('Training Set Biased Ratio')

            os.makedirs(os.path.join(args.plot_dir, plot_folder), exist_ok=True)
            save_path = os.path.join(args.plot_dir, plot_folder, '_'.join([title, 'static.png']))
            plt.savefig(save_path)

    for view in config['log']['view']:
        for res in config['log']['res']:
            for i, biased_ratio in enumerate(biased_ratios):
                plot_folder = '_'.join(['mixed', view, res, str(biased_ratio)])
                log_folder = '_'.join(['mixed', view, res, str(biased_ratio)])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                train_losses = result['tl']
                train_accs = result['ta']
                val_losses = result['vl'][i]
                val_accs = result['va'][i]

                epochs = range(1, len(train_losses) + 1)
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()

                ax1.plot(epochs, train_losses, label='Train loss', color='blue')
                ax1.plot(epochs, val_losses, label='Val loss', color='red')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_ylim([config['plot']['loss_min'], config['plot']['loss_max']])
                ax1.legend(loc='upper left')

                ax2.plot(epochs, train_accs, label='Train acc', color='green')
                ax2.plot(epochs, val_accs, label='Val acc', color='orange')
                ax2.set_ylabel('Accuracy')
                ax2.set_ylim([config['plot']['acc_min'], config['plot']['acc_max']])
                ax2.legend(loc='upper right')

                ax1.legend(loc='lower right')
                ax2.legend(loc='upper right')
                plt.tight_layout()
                title = '_'.join([model_name, plot_folder])
                # plt.title(title)

                os.makedirs(os.path.join(args.plot_dir, plot_folder), exist_ok=True)
                save_path = os.path.join(args.plot_dir, plot_folder, '_'.join([title, 'loss_acc.png']))
                plt.savefig(save_path)


if __name__ == '__main__':
    plot_mixed_loss_acc()
    print('Done')
