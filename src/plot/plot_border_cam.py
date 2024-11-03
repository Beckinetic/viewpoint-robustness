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
cue_conflict = config['log']['eval_views']
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
cam_types = config['plot']['cam_types']


def plot_border_cam():
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    fig.subplots_adjust(left=0.1, right=0.7, wspace=0.3)

    # plot individual cams
    for ind_cam, cam_type in enumerate(cam_types):
        # counts_array = np.zeros((len(views) * len(backgrounds), 10))
        cam_data_per_view = []
        for ind_background, background in enumerate(backgrounds):
            for ind_view, view in enumerate(views):
                log_folder = '_'.join([background, view, res])
                result_path = os.path.join(log_dir, log_folder, 'activations', str(max_epoch), view,
                                           '_'.join([cam_type, 'cams.pkl']))
                with open(result_path, 'rb') as f:
                    cams = pickle.load(f)

                mean_cam_per_img = np.array([np.mean(item) for item in cams])
                mean_cam_per_img = mean_cam_per_img[~np.isnan(mean_cam_per_img)]
                # bins = np.arrange(0, 1.1, 0.1)  # Bins from 0 to 1, in steps of 0.1
                # counts, bin_edges = np.histogram(mean_cam_per_img, bins=bins)
                # counts_array[ind_view] = counts
                cam_data_per_view.append(mean_cam_per_img)

        cam_data_per_view = np.array(cam_data_per_view)

        ax = axes[0, ind_cam]
        bplot = ax.boxplot(cam_data_per_view, labels=view_plot_name, patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlim([0.1, len(views) + 0.9])
        ax.set_ylim([0, 1.1])
        ax.set_title(cam_type.capitalize() + ' CAM')
        if ind_cam == 0:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_xticklabels([])
            ax.set_ylabel('CAM Value')
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    for ind_cam, cam_type in enumerate(cam_types):
        cam_data_per_view = []
        for ind_background, background in enumerate(backgrounds):
            for ind_view, view in enumerate(views):
                log_folder = '_'.join([background, view, res])
                other_views = [item for item in views if item != view]
                mean_cam_per_img = []
                for other_view in other_views:
                    result_path = os.path.join(log_dir, log_folder, 'activations', str(max_epoch), other_view,
                                               '_'.join([cam_type, 'cams.pkl']))
                    with open(result_path, 'rb') as f:
                        cams = pickle.load(f)

                    mean_cam_per_img.extend([np.mean(item) for item in cams])

                mean_cam_per_img = np.array(mean_cam_per_img)
                mean_cam_per_img = mean_cam_per_img[~np.isnan(mean_cam_per_img)]
                # bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1, in steps of 0.1
                # counts, bin_edges = np.histogram(mean_cam_per_img, bins=bins)
                # counts_array[ind_view] = counts
                cam_data_per_view.append(mean_cam_per_img)

        ax = axes[1, ind_cam]
        # ax.stackplot(np.arange(0, 1, 0.1), counts_array, labels=view_plot_name, colors=palette.values(), alpha=0.6)
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 124000])
        bplot = ax.boxplot(cam_data_per_view, labels=view_plot_name, patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlim([0.1, len(views) + 0.9])
        ax.set_ylim([0, 1.1])
        if ind_cam == 0:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_ylabel('CAM Value')
        else:
            ax.set_yticklabels([])

        if ind_cam == 1:
            ax.set_xlabel('CAM Value')

    color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                    for color in palette.values()]
    axes[1, len(cam_types) - 1].legend(color_legend, view_plot_name, loc=(1.1, 0))

    # save the plot
    save_path = os.path.join(plot_dir, 'border_cam.png')
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_border_cam()
    print('Done')
