import argparse
import os
import pickle

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib.patches import Patch

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
            print(f"train_acc of {view}: {train_acc[max_epoch-1]}")

            # plot accuracy
            ax1.plot(epochs, train_acc, color=palette[view])
            ax1.set_xlabel('Training Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.set_ylim([acc_min, acc_max])
            ax1.set_title("Training Accuracy")

            ax1.grid(visible=False)
            # ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

            # create color legend
            color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                            for color in colors]
            ax1.legend(color_legend, view_plot_name, loc=(1.06, 0.15))

    # save the plot
    save_path = os.path.join(plot_dir, f'{model_name}_train_acc_loss.svg')
    plt.savefig(save_path, format='svg')
    plt.close(fig)


def plot_val_acc_loss():
    # plot validation accuracy and loss
    # set plotting canvas
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(14, 4))
    fig.subplots_adjust(top=0.85)

    # plot the validation accuracy and loss on the same view images (Figure 1-B)
    for background in backgrounds:
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
            ax0.plot(epochs, train_acc, color=palette[view])
            ax0.set_xlabel('Training Epoch')
            ax0.set_ylabel('Accuracy (%)\nClean images', fontsize=12)
            ax0.set_ylim([acc_min, acc_max])
            ax0.set_title("Training Accuracy")
            ax0.spines[['top', 'right']].set_visible(False)

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
            print(f"validation_acc of {view}: {validation_acc[max_epoch - 1]}")

            # plot accuracy
            ax1.plot(epochs, validation_acc, color=palette[view], linewidth=2)
            ax1.set_xlabel('Training Epoch')
            ax1.set_yticklabels([])
            ax1.set_ylim([acc_min, acc_max])
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.set_title("Test Accuracy on ID Viewpoints")

    # plot the accuracy and loss when tested on viewpoint o.o.d. test set
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
                print(f"test_acc of {view}: {test_acc[max_epoch - 1]}")

                # plot accuracy
                ax2.plot(epochs, test_acc, color=palette[view], linewidth=2)
                ax2.set_xlabel('Training Epoch')
                ax2.set_ylim([acc_min, acc_max])
                ax2.set_yticklabels([])
                ax2.spines[['top', 'right']].set_visible(False)
                ax2.set_title("Testing Accuracy on OOD Viewpoints")

            # create color legend
            color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                            for color in colors]
            ax1.legend(color_legend, view_plot_name, loc=(-0.45, 1.1), ncols=4)

    # plot the validation accuracy and loss on the same view images (Figure 1-B)
    val_accs = []
    test_accs = []
    for background in backgrounds:
        for ind_view, view in enumerate(views):
            # get validation results of current model
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
            ood_result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'test', 'data.pkl']))
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            with open(ood_result_path, 'rb') as f:
                ood_result = pickle.load(f)
            test_view = test_views[0]
            validation_acc = result['va']
            test_acc = ood_result['tea']

            last_epoch_validation_acc = validation_acc[view][max_epoch - 1]
            last_epoch_test_acc = test_acc[test_view][max_epoch - 1]
            val_accs.append(last_epoch_validation_acc)
            test_accs.append(last_epoch_test_acc)

    # plot accuracy as bar plot
    # Bar positions
    bar_width = 0.35
    x = np.arange(len(views))

    # Plot bars
    for i, view in enumerate(views):
        # ID (validation) — outline only
        ax3.bar(
            x[i] - 0.01,
            val_accs[i],
            bar_width,
            label=f'Validation ({view_plot_name[i]})',
            color='none',
            edgecolor=palette[view],
            linewidth=1.8
        )
        # OOD (test) — solid fill
        ax3.bar(
            x[i] + bar_width + 0.01,
            test_accs[i],
            bar_width,
            label=f'Test ({view_plot_name[i]})',
            color=palette[view]
        )


    # Customize plot
    ax3.set_xlabel('View')
    ax3.set_yticklabels([])
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.set_title('Last Epoch Test Accuracy')
    ax3.set_xticks(x + bar_width / 2)
    short_view_plot_name = ["Fixed", "Extra R.", "Restricted", "Full"]
    ax3.set_xticklabels(short_view_plot_name)
    ax3.set_ylim(acc_min, acc_max)

    legend_elements = [
        Patch(facecolor='none', edgecolor='gray', linewidth=1.8, label='ID'),
        Patch(facecolor='gray', edgecolor='gray', label='OOD')
    ]
    ax3.legend(handles=legend_elements, loc=(0.15, 1.1), ncols=2)

    # ax0.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax2.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax3.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    ax0.grid(visible=False)
    ax1.grid(visible=False)
    ax2.grid(visible=False)
    ax3.grid(visible=False)
    # Add chance-level (1/32 ≈ 3.125%) reference line to all subplots
    chance_acc = 100.0 / 32.0  # convert to percentage for y-axis
    for ax in (ax0, ax1, ax2, ax3):
        ax.axhline(y=chance_acc, linestyle='--', linewidth=1, color='gray', alpha=0.7, zorder=0)

    # save the plot
    save_path = os.path.join(plot_dir, f'{model_name}_validation_acc_loss.svg')
    plt.savefig(save_path, format='svg')
    plt.close(fig)


def plot_last_epoch_acc():
    fig, (ax0, ax1, ax3) = plt.subplots(1, 3, figsize=(14, 4))  # slightly reduced height
    fig.subplots_adjust(top=0.85)
    fig.delaxes(ax1)  # Remove ax1
    fig.delaxes(ax3)  # Remove ax2

    # # plot the validation accuracy and loss on the same view images (Figure 1-B)
    # val_accs = []
    # test_accs = []
    # for background in backgrounds:
    #     for ind_view, view in enumerate(views):
    #         # get validation results of current model
    #         log_folder = '_'.join([background, view, res])
    #         result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
    #         ood_result_path = os.path.join(log_dir, log_folder, '_'.join([model_name, 'test', 'data.pkl']))
    #         with open(result_path, 'rb') as f:
    #             result = pickle.load(f)
    #         with open(ood_result_path, 'rb') as f:
    #             ood_result = pickle.load(f)
    #         test_view = test_views[0]
    #         validation_acc = result['va']
    #         test_acc = ood_result['tea']
    #
    #         last_epoch_validation_acc = validation_acc[view][max_epoch-1]
    #         last_epoch_test_acc = test_acc[test_view][max_epoch-1]
    #         val_accs.append(last_epoch_validation_acc)
    #         test_accs.append(last_epoch_test_acc)
    #
    # # plot accuracy as bar plot
    # # Bar positions
    # bar_width = 0.35
    # x = np.arange(len(views))
    #
    # # Plot bars
    # for i, view in enumerate(views):
    #     ax0.bar(x[i] - 0.01, val_accs[i], bar_width, label=f'Validation ({view_plot_name[i]})', color=palette[view])
    #     ax0.bar(x[i] + bar_width + 0.01, test_accs[i], bar_width, label=f'Test ({view_plot_name[i]})', color=palette[view], hatch='//')
    #
    # # Customize plot
    # ax0.set_xlabel('View')
    # ax0.set_yticklabels([])
    # ax0.spines[['top', 'right']].set_visible(False)
    # ax0.set_title('Last Epoch Test Accuracy')
    # ax0.set_xticks(x + bar_width / 2)
    # short_view_plot_name = ["Fixed", "Extra R.", "Restricted", "Full"]
    # ax0.set_xticklabels(short_view_plot_name)
    # ax0.set_ylim(acc_min, acc_max)
    #
    # legend_elements = [
    #     Patch(facecolor='gray', label='ID'),
    #     Patch(facecolor='gray', hatch='//', label='OOD')
    # ]
    # ax0.legend(handles=legend_elements, loc=(-0.5, 1.2), bbox_to_anchor=(1.05, 1), ncols=1)
    #
    # # save the plot
    # save_path = os.path.join(plot_dir, f'{model_name}_last_epoch_accuracy.svg')
    # plt.savefig(save_path, format='svg')
    # plt.close(fig)


if __name__ == '__main__':
    plot_train_acc_loss()
    plot_val_acc_loss()
    plot_last_epoch_acc()
    print('Done')
