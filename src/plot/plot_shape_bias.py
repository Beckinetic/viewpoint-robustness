import argparse
import os
import pickle
import yaml
from matplotlib import pyplot as plt, lines
import numpy as np

from src.static import category_labels

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


def get_decision_proportion_and_shape_bias(shape_decisions, texture_decisions, all_decisions):
    # sum up decisions
    shape_decision_count = sum(shape_decisions.values())
    texture_decision_count = sum(texture_decisions.values())
    shape_and_texture_decision_count = shape_decision_count + texture_decision_count
    all_decisions_count = sum(all_decisions.values())

    # compute decision proportions and shape bias
    shape_and_texture_decision_proportion = shape_and_texture_decision_count / all_decisions_count
    shape_decision_proportion = shape_decision_count / all_decisions_count
    texture_decision_proportion = texture_decision_count / all_decisions_count
    shape_bias = np.sqrt(shape_decision_count / shape_and_texture_decision_count) * np.sqrt(
        shape_decision_count / all_decisions_count)  # accuracy corrected shape bias
    # texture_bias = np.sqrt(total_texture_decision_count / shape_and_texture_decision_count) * np.sqrt(
    #     total_texture_decision_count / all_decisions_count)
    return shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion, shape_bias


def get_standard_errors(shape_decisions, texture_decisions, all_decisions):
    shape_decisions_se = ((np.std(np.array(list(shape_decisions.values())) / np.array(list(all_decisions.values())))) /
                          np.sqrt(len(shape_decisions.values())))
    texture_decisions_se = ((np.std(np.array(list(texture_decisions.values())) / np.array(list(all_decisions.values())))) /
                            np.sqrt(len(texture_decisions.values())))
    shape_and_texture_decisions_se = (np.std((np.array(list(texture_decisions.values()))+np.array(list(shape_decisions.values())))
                                             / np.array(list(all_decisions.values()))) /
                                      np.sqrt(len(texture_decisions.values())))
    shape_bias_by_category = {label: [] for label in category_labels.keys()}
    for category in shape_decisions:
        total_decisions = shape_decisions[category] + texture_decisions[category]
        if total_decisions > 0:
            shape_bias_by_category[category].append(np.sqrt(shape_decisions[category] / total_decisions) * np.sqrt(
                shape_decisions[category] / all_decisions[category]))
        else:
            shape_bias_by_category[category].append(0)
    shape_bias_se = np.std(np.array(list(shape_bias_by_category.values())))/np.sqrt(len(shape_bias_by_category.values()))
    return shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se


def plot_shape_bias():
    # plot for the decision proportions and shape bias (Figure 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.subplots_adjust(left=0.1, right=0.7, wspace=0.3)
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            # get cue-conflict results for current model
            if pasted:
                result_file_name = 'shape_bias_pasted.pkl'
            else:
                result_file_name = 'shape_bias.pkl'
            log_folder = '_'.join([background, view, res])
            cue_conflict_folder = '_'.join(['cue_conflict', view]) # get cue-conflict dataset of matched-view
            result_path = os.path.join(log_dir, log_folder, cue_conflict_folder, result_file_name)
            with open(result_path, 'rb') as file:
                results = pickle.load(file)
            content_accuracy = results['content_accuracies'][0]
            shape_decisions = results['shape_decisions'][0]
            texture_decisions = results['texture_decisions'][0]
            all_decisions = results['total_decisions'][0]

            shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion, shape_bias =\
                get_decision_proportion_and_shape_bias(shape_decisions, texture_decisions, all_decisions)
            shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se = (
                get_standard_errors(shape_decisions, texture_decisions, all_decisions))

            # plot the decision proportions and shape bias
            # set up bar width and positions
            bar_width = 0.2

            # set a flattened index
            ind_flattened = ind_background * len(view) + ind_view

            # Plot bars for texture/shape decision proportions on primary y-axis
            ax1.bar(ind_flattened - 1.5 * bar_width, shape_and_texture_decision_proportion,
                    yerr=shape_and_texture_decisions_se,
                    width=bar_width,
                    label='Shape + Texture Decision Proportion',
                    color=palette_shape_bias['shape_and_texture_proportion'])
            ax1.bar(ind_flattened - 0.5 * bar_width, shape_decision_proportion,
                    yerr=shape_decisions_se,
                    width=bar_width,
                    label='Texture Decision Proportion',
                    color=palette_shape_bias['shape_proportion'])
            ax1.bar(ind_flattened + 0.5 * bar_width, texture_decision_proportion,
                    yerr=texture_decisions_se,
                    width=bar_width,
                    label='Shape Decision Proportion',
                    color=palette_shape_bias['texture_proportion'])

            ax1_twin.bar(ind_flattened + 1.5 * bar_width + 0.05, shape_bias, width=bar_width, yerr=shape_bias_se,
                         label='Shape Bias', color=palette_shape_bias['shape_bias'], hatch="//")

    # ax1 settings
    ax1.set_ylabel('Decision Proportion')
    ax1_twin.set_ylabel('Shape Bias')
    ax1.set_ylim([decision_proportion_min, decision_proportion_max])
    ax1_twin.set_ylim([shape_bias_min, shape_bias_max])
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.title.set_text('Viewpoint Distribution Matched')

    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            # get cue-conflict results for current model
            if pasted:
                result_file_name = 'shape_bias_pasted.pkl'
            else:
                result_file_name = 'shape_bias.pkl'
            log_folder = '_'.join([background, view, res])

            # get cue-conflict dataset of non-matched-view
            shape_and_texture_decision_proportions = []
            shape_decision_proportions = []
            texture_decision_proportions = []
            shape_biases = []
            other_views = [item for item in views if item != view]
            shape_decisions_merged = {label: 0 for label in category_labels.keys()}
            texture_decisions_merged = {label: 0 for label in category_labels.keys()}
            all_decisions_merged = {label: 0 for label in category_labels.keys()}
            for other_view in other_views:
                cue_conflict_folder = '_'.join(['cue_conflict', other_view])
                result_path = os.path.join(log_dir, log_folder, cue_conflict_folder, result_file_name)
                with open(result_path, 'rb') as file:
                    results = pickle.load(file)
                content_accuracy = results['content_accuracies'][0]
                shape_decisions = results['shape_decisions'][0]
                texture_decisions = results['texture_decisions'][0]
                all_decisions = results['total_decisions'][0]
                shape_decisions_merged = {key: shape_decisions_merged[key] + shape_decisions[key]
                                          for key in shape_decisions_merged}
                texture_decisions_merged = {key: texture_decisions_merged[key] + texture_decisions[key]
                                            for key in texture_decisions_merged}
                all_decisions_merged = {key: all_decisions_merged[key] + all_decisions[key]
                                        for key in all_decisions_merged}
                (shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion,
                 shape_bias) = get_decision_proportion_and_shape_bias(shape_decisions, texture_decisions, all_decisions)
                shape_and_texture_decision_proportions.append(shape_and_texture_decision_proportion)
                shape_decision_proportions.append(shape_decision_proportion)
                texture_decision_proportions.append(texture_decision_proportion)
                shape_biases.append(shape_bias)
            shape_and_texture_decision_proportion = np.mean(shape_and_texture_decision_proportions, axis=0)
            shape_decision_proportion = np.mean(shape_decision_proportions, axis=0)
            texture_decision_proportion = np.mean(texture_decision_proportions, axis=0)
            shape_bias = np.mean(shape_biases, axis=0)
            shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se = (
                get_standard_errors(shape_decisions_merged, texture_decisions_merged, all_decisions_merged))

            # plot the decision proportions and shape bias
            # set up bar width and positions
            bar_width = 0.2

            # set a flattened index
            ind_flattened = ind_background * len(view) + ind_view

            # Plot bars for texture/shape decision proportions on primary y-axis
            ax2.bar(ind_flattened - 1.5 * bar_width, shape_and_texture_decision_proportion,
                    yerr=shape_and_texture_decisions_se,
                    width=bar_width,
                    label='Shape + Texture Decision Proportion',
                    color=palette_shape_bias['shape_and_texture_proportion'])
            ax2.bar(ind_flattened - 0.5 * bar_width, shape_decision_proportion,
                    yerr=shape_decisions_se,
                    width=bar_width,
                    label='Texture Decision Proportion',
                    color=palette_shape_bias['shape_proportion'])
            ax2.bar(ind_flattened + 0.5 * bar_width, texture_decision_proportion,
                    yerr=texture_decisions_se,
                    width=bar_width,
                    label='Shape Decision Proportion',
                    color=palette_shape_bias['texture_proportion'])

            ax2_twin.bar(ind_flattened + 1.5 * bar_width + 0.05, shape_bias, yerr=shape_bias_se, width=bar_width,
                         label='Shape Bias', color=palette_shape_bias['shape_bias'], hatch="//")

    ax2.set_xticks(np.arange(len(backgrounds) * len(views)))
    ax2.set_xticklabels([item+' model' for item in view_plot_name])

    # ax2 settings
    ax2.set_ylabel('Decision Proportion')
    ax2.set_xlabel('Models')
    ax2.set_ylim([decision_proportion_min, decision_proportion_max])
    ax2_twin.set_ylabel('Shape Bias')
    ax2_twin.set_ylim([shape_bias_min, shape_bias_max])
    ax2.title.set_text('Viewpoint Distribution Non-Matched')
    # create color legend
    color_legend = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                    for color in palette_shape_bias.values()]
    ax2.legend(color_legend, shape_bias_metric, loc=(1.1, 0))
    # save the plot
    save_path = os.path.join(plot_dir, 'shape_bias.png')
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_shape_bias()
    print('Done')
