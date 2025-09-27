import argparse
import os
import pickle

import numpy as np
import yaml
from matplotlib import pyplot as plt, lines

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
res = config['log']['res']
backgrounds = config['log']['background']
eval_views = config['log']['eval']['view']
eval_ood_views = config['log']['eval']['ood_view']
if config['log']['eval']['suffix']:
    suffix = config['log']['eval']['suffix']
loss_min = config['plot']['loss_min']
loss_max = config['plot']['loss_max']
acc_min = config['plot']['acc_min']
acc_max = config['plot']['acc_max']
loss_change_min = config['plot']['loss_change_min']
loss_change_max = config['plot']['loss_change_max']
acc_change_min = config['plot']['acc_change_min']
acc_change_max = config['plot']['acc_change_max']
decision_proportion_min = config['plot']['decision_proportion_min']
decision_proportion_max = config['plot']['decision_proportion_max']
shape_bias_min = config['plot']['shape_bias_min']
shape_bias_max = config['plot']['shape_bias_max']
palette = config['plot']['palette']
palette_shape_bias = config['plot']['palette_shape_bias']
colors = list(palette.values())
view_plot_name = config['plot']['view']
shape_bias_metric = config['plot']['shape_bias_metric']
distortion_types = config['plot']['distortion_types']
max_severity = config['plot']['max_severity']


def plot_robustness():
    # plot for the accuracy on distorted pictures (Figure 2)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 5))
    fig.subplots_adjust(wspace=0.2)

    # plot for robustness on the distorted images of the matched view
    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            log_folder = '_'.join([background, view, res])

            # load result
            result_path = os.path.join(log_dir, log_folder, 'robustness.pkl')
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            content_acc = result['ca']
            distortion_acc = result['da']

            # obtain mean accuracy and standard error
            accuracy = np.zeros((len(distortion_types), max_severity + 1))
            for ind_distortion, distortion_type in enumerate(distortion_types):
                accuracy[ind_distortion, 0] = content_acc[view]
                accuracy[ind_distortion, 1: max_severity + 1] = list(distortion_acc[view][distortion_type].values())
                # accuracy[ind_distortion, :] = accuracy[ind_distortion, :] / content_acc[view] * 100
            mean_accuracy = np.mean(accuracy, axis=0)
            print(f"mean_accuracy (IDV) of {view}: {mean_accuracy}")
            se_accuracy = np.std(accuracy, axis=0, ddof=1) / np.sqrt(accuracy.shape[0])

            severities = range(0, max_severity + 1)
            ax1.plot(severities, mean_accuracy, color=palette[view], label=view_plot_name[ind_view])
            ax1.fill_between(severities, mean_accuracy - se_accuracy, mean_accuracy + se_accuracy,
                             color=palette[view], alpha=0.2)
            ax1.set_xlabel('Severity')
            ax1.set_ylabel('Accuracy (%)\nCorrupted Images', fontsize=12)
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.set_xticks(severities)
            ax1.set_ylim(acc_min, acc_max)
            ax1.title.set_text('Accuracy on Corrupted Images (IDV)')

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            log_folder = '_'.join([background, view, res])
            ood_views = eval_ood_views
            accuracy = np.zeros((len(distortion_types), max_severity + 1, len(ood_views)))
            for ind_ood_view, ood_view in enumerate(ood_views):
                # load result
                result_path = os.path.join(log_dir, log_folder, 'robustness.pkl')
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                content_acc = result['ca']
                distortion_acc = result['da']

                # obtain mean accuracy and standard error
                for ind_distortion, distortion_type in enumerate(distortion_types):
                    accuracy[ind_distortion, 0, ind_ood_view] = content_acc[ood_view]
                    accuracy[ind_distortion, 1: max_severity + 1, ind_ood_view] = list(distortion_acc[ood_view]
                                                                                         [distortion_type].values())

            accuracy = np.mean(accuracy, axis=2)  # average results on non-matched views
            mean_accuracy = np.mean(accuracy, axis=0)
            print(f"mean_accuracy (OOD) of {view}: {mean_accuracy}")
            se_accuracy = np.std(accuracy, axis=0, ddof=1) / np.sqrt(accuracy.shape[0])

            severities = range(0, max_severity + 1)
            ax3.plot(severities, mean_accuracy, color=palette[view], label=view_plot_name[ind_view])
            ax3.fill_between(severities, mean_accuracy - se_accuracy, mean_accuracy + se_accuracy,
                             color=palette[view], alpha=0.2)
            ax3.set_xlabel('Severity')
            ax3.set_xticks(severities)
            ax3.set_yticklabels([])
            ax3.set_ylim(acc_min, acc_max)
            ax3.title.set_text('Accuracy on Corrupted Images (HOV)')
            ax3.spines[['top', 'right']].set_visible(False)

            color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                            for color in palette.values()]
            ax3.legend(color_legend, view_plot_name, loc=(-1.15, 1.08), ncols=4)

    # ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax3.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    ax1.grid(visible=False)
    ax3.grid(visible=False)

    chance_acc = 100.0 / 32.0  # convert to percentage for y-axis
    for ax in (ax1, ax3):
        ax.axhline(y=chance_acc, linestyle='--', linewidth=1, color='gray', alpha=0.7, zorder=0)

    # save the plot
    save_path = os.path.join(plot_dir, f'{model_name}_robustness.svg')
    plt.savefig(save_path, format='svg')
    plt.close(fig)


if __name__ == '__main__':
    plot_robustness()
    print('Done')
