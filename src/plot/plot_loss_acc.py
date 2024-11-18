import argparse
import os
import pickle
import warnings

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import lines

# set rc parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = '500'  # medium weight
plt.rcParams['font.stretch'] = 'semi-expanded'  # slightly expanded
plt.rcParams['figure.dpi'] = 300

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--plot-dir', type=str, default='plots/')

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f.read())
log_dir = args.log_dir
plot_dir = args.plot_dir

model_name = config['model']['model_name']
max_epoch = config['model']['max_epoch']
epochs = range(1, max_epoch + 1)
views = config['log']['view']
test_views = config['log']['test_view']
res = config['log']['res']
backgrounds = config['log']['background']
loss_min = config['plot']['loss_min']
loss_max = config['plot']['loss_max']
acc_min = config['plot']['acc_min']
acc_max = config['plot']['acc_max']
loss_change_min = config['plot']['loss_change_min']
loss_change_max = config['plot']['loss_change_max']
acc_change_min = config['plot']['acc_change_min']
acc_change_max = config['plot']['acc_change_max']
palette = config['plot']['palette']
colors = list(palette.values())
view_plot_name = config['plot']['view']


def plot_train_acc_loss():
    # plot training accuracies and loss by models (Figure 1-A)
    # set plotting canvas
    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.1, right=0.8)

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            # get training results of current model
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            train_loss = result['tl']
            train_acc = result['ta']

            # cut data by max epoch
            train_loss = train_loss[0:max_epoch]
            train_acc = train_acc[0:max_epoch]

            # plot accuracy
            ax1.plot(epochs, train_acc, color=palette[view])
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_ylim([acc_min, acc_max])
            ax1.set_title("Training Accuracy")

            ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

            # create color legend
            color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                            for color in colors]
            ax1.legend(color_legend, view_plot_name, loc=(1.06, 0.15))

    # save the plot
    save_path = os.path.join(plot_dir, 'train_acc_loss.png')
    plt.savefig(save_path)
    plt.close(fig)


def plot_val_acc_loss():
    # plot validation accuracy and loss
    # set plotting canvas
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(left=0.1, right=0.8, wspace=0.2)

    # plot the validation accuracy and loss on the same view images (Figure 1-B)
    for background in backgrounds:
        for ind_view, view in enumerate(views):
            # get validation results of current model
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            validation_loss = result['vl']
            validation_acc = result['va']

            # cut data by max epoch
            validation_loss = validation_loss[view][0:max_epoch]
            validation_acc = validation_acc[view][0:max_epoch]

            # plot accuracy
            ax1.plot(epochs, validation_acc, color=palette[view], linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_ylim([acc_min, acc_max])
            ax1.set_title("Test Accuracy on In-distribution Viewpoint Data")

            # global plot settings
            ax1.grid(visible=True, linestyle='--', linewidth=1, color='gray', alpha=0.6)

    # # plot the accuracy and loss when validated on non-matched view images
    # for background in backgrounds:
    #     for ind_view, view in enumerate(views):
    #         # get validation results of current model
    #         log_folder = '_'.join([background, view, res])
    #         result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
    #         with open(result_path, 'rb') as f:
    #             result = pickle.load(f)
    #         validation_loss = result['vl']
    #         validation_acc = result['va']
    #
    #         other_views = [item for item in views if item != view]
    #         validation_loss_other_view = []
    #         validation_acc_other_view = []
    #         for other_view in other_views:
    #             validation_loss_other_view.append(np.array(validation_loss[other_view][0:max_epoch]))
    #             validation_acc_other_view.append(np.array(validation_acc[other_view][0:max_epoch]))
    #         mean_val_loss_other_view = np.mean(validation_loss_other_view, axis=0)
    #         mean_val_acc_other_view = np.mean(validation_acc_other_view, axis=0)
    #
    #         # plot accuracy
    #         ax2.plot(epochs, mean_val_acc_other_view, color=palette[view])
    #         ax2.set_xlabel('Epoch')
    #         # ax2.set_ylabel('Accuracy (%)')
    #         ax2.set_ylim([acc_min, acc_max])
    #         ax2.set_title("Mean Validation Accuracy on Viewpoint Non-matched Datasets")
    #
    #         # global plot settings
    #         ax2.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

    # plot the accuracy and loss when tested on viewpoint o.o.d. test set (except for full view dataset)
    for background in backgrounds:
        for ind_view, view in enumerate(views):
            # get validation results of current model
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'test', 'data.pkl']))
            with open(result_path, 'rb') as f:
                result = pickle.load(f)

            for test_view in test_views:
                test_loss = result['tel'][test_view][0:max_epoch]
                test_acc = result['tea'][test_view][0:max_epoch]

                # plot accuracy
                ax3.plot(epochs, test_acc, color=palette[view], linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylim([acc_min, acc_max])
                ax3.set_yticks([])
                ax3.set_title("Testing Accuracy on OOD Viewpoint Data")

                # global plot settings
                ax3.grid(visible=True, linestyle='--', linewidth=1, color='gray', alpha=0.6)

                # create color legend
                color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                                for color in colors]
                ax3.legend(color_legend, view_plot_name, loc=(1.1, 0.15))

    # save the plot
    save_path = os.path.join(plot_dir, 'validation_acc_loss.png')
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_train_acc_loss()
    plot_val_acc_loss()
    print('Done')
