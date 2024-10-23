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
cue_conflict = config['log']['cue_conflict']
pasted = config['log']['pasted']
backgrounds = config['log']['background']
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.1, right=0.7, wspace=0.3)

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            log_folder = '_'.join([background, view, res])
            distortion_folder = '_'.join(['ood', view])  # they are actually NOT o.o.d., but I didn't realise that

            # load result
            result_file_name = 'distortion_acc.pkl'
            content_acc_file_name = 'content_acc.pkl'
            result_path = os.path.join(log_dir, log_folder, 'robustness', str(max_epoch), distortion_folder,
                                       result_file_name)
            content_acc_path = os.path.join(log_dir, log_folder, 'robustness', str(max_epoch), distortion_folder,
                                            content_acc_file_name)
            with open(result_path, 'rb') as f:
                distortion_acc = pickle.load(f)
            with open(content_acc_path, 'rb') as f:
                content_acc = pickle.load(f)

            # obtain mean accuracy and standard error
            accuracy = np.zeros((len(distortion_types), max_severity + 1))
            for ind_distortion, distortion_type in enumerate(distortion_types):
                accuracy[ind_distortion, 0] = content_acc[str(max_epoch)]
                accuracy[ind_distortion, 1: max_severity + 1] = list(distortion_acc[str(max_epoch)][distortion_type].values())
            accuracy = np.multiply(accuracy, 100)
            mean_accuracy = np.mean(accuracy, axis=0)
            se_accuracy = np.std(accuracy, axis=0, ddof=1) / np.sqrt(accuracy.shape[0])

            severities = range(0, max_severity + 1)
            ax1.plot(severities, mean_accuracy, color=palette[view], label=view_plot_name[ind_view])
            ax1.fill_between(severities, mean_accuracy - se_accuracy, mean_accuracy + se_accuracy,
                             color=palette[view], alpha=0.2)
            ax1.set_xlabel('Severity')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xticks(severities)
            ax1.set_ylim(acc_min, acc_max)
            ax1.title.set_text('Viewpoint Matched')
            ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            log_folder = '_'.join([background, view, res])
            other_views = [item for item in views if item != view]
            accuracy = np.zeros((len(distortion_types), max_severity + 1, len(other_views)))
            for ind_other_view, other_view in enumerate(other_views):
                distortion_folder = '_'.join(
                    ['ood', other_view])  # they are actually NOT o.o.d., but I didn't realise that

                # load result
                result_file_name = 'distortion_acc.pkl'
                content_acc_file_name = 'content_acc.pkl'
                result_path = os.path.join(log_dir, log_folder, 'robustness', str(max_epoch), distortion_folder,
                                           result_file_name)
                content_acc_path = os.path.join(log_dir, log_folder, 'robustness', str(max_epoch), distortion_folder,
                                                content_acc_file_name)
                with open(result_path, 'rb') as f:
                    distortion_acc = pickle.load(f)
                with open(content_acc_path, 'rb') as f:
                    content_acc = pickle.load(f)

                # obtain mean accuracy and standard error
                for ind_distortion, distortion_type in enumerate(distortion_types):
                    accuracy[ind_distortion, 0, ind_other_view] = content_acc[str(max_epoch)]
                    accuracy[ind_distortion, 1: max_severity + 1, ind_other_view] = list(distortion_acc[str(max_epoch)]
                                                                                         [distortion_type].values())

            accuracy = np.multiply(accuracy, 100)
            accuracy = np.mean(accuracy, axis=2)  # average results on non-matched views
            mean_accuracy = np.mean(accuracy, axis=0)
            se_accuracy = np.std(accuracy, axis=0, ddof=1) / np.sqrt(accuracy.shape[0])

            severities = range(0, max_severity + 1)
            ax2.plot(severities, mean_accuracy, color=palette[view], label=view_plot_name[ind_view])
            ax2.fill_between(severities, mean_accuracy - se_accuracy, mean_accuracy + se_accuracy,
                             color=palette[view], alpha=0.2)
            ax2.set_xlabel('Severity')
            ax2.set_xticks(severities)
            ax2.set_ylim(acc_min, acc_max)
            ax2.title.set_text('Viewpoint Non-Matched')
            ax2.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

            color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                            for color in palette.values()]
            ax2.legend(color_legend, view_plot_name, loc=(1.1, 0))

    # save the plot
    save_path = os.path.join(plot_dir, 'robustness.png')
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_robustness()
    print('Done')
