import argparse
import os
import pickle

import pandas as pd
import yaml
from matplotlib import pyplot as plt, lines
import seaborn as sns
import numpy as np
# from statannotations.Annotator import Annotator

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
backgrounds = config['log']['background']
eval_views = config['log']['eval']['view']
ood_view = config['log']['eval']['ood_view']
suffixes = ['']
if config['log']['eval']['suffix'] is not None:
    suffixes.extend(config['data']['eval']['suffix'])
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
    # shape_bias = np.sqrt(shape_decision_count / shape_and_texture_decision_count) * np.sqrt(
    #     shape_decision_count / all_decisions_count)  # accuracy corrected shape bias
    shape_bias = shape_decision_count / shape_and_texture_decision_count

    return shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion, shape_bias


def get_standard_errors(shape_decisions, texture_decisions, all_decisions):
    shape_decisions_se = ((np.std(np.array(list(shape_decisions.values())) / np.array(list(all_decisions.values())))) /
                          np.sqrt(len(shape_decisions.values())))
    texture_decisions_se = (
                (np.std(np.array(list(texture_decisions.values())) / np.array(list(all_decisions.values())))) /
                np.sqrt(len(texture_decisions.values())))
    shape_and_texture_decisions_se = (
                np.std((np.array(list(texture_decisions.values())) + np.array(list(shape_decisions.values())))
                       / np.array(list(all_decisions.values()))) /
                np.sqrt(len(texture_decisions.values())))
    shape_bias_by_category = {label: [] for label in category_labels.keys()}
    for category in shape_decisions:
        total_decisions = shape_decisions[category] + texture_decisions[category]
        if total_decisions > 0:
            shape_bias_by_category[category].append(np.sqrt(shape_decisions[category] / total_decisions) * np.sqrt(
                shape_decisions[category] / all_decisions[category]))
            # shape_bias_by_category[category].append(np.sqrt(shape_decisions[category] / total_decisions))
        else:
            shape_bias_by_category[category].append(0)
    shape_bias_se = np.std(np.array(list(shape_bias_by_category.values()))) / np.sqrt(
        len(shape_bias_by_category.values()))
    return shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se


def plot_shape_bias(suffix):
    # plot for the decision proportions and shape bias (Figure 2)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    width_factor_decisions = 0.9
    width_factor_shape_bias = 0.8

    # plot shape bias on viewpoint matched datasets
    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            # get cue-conflict results for current model
            result_file_name = 'shape_bias.pkl'
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, result_file_name)
            with open(result_path, 'rb') as file:
                results = pickle.load(file)
            if suffix:
                key = '_'.join([view, suffix])
            else:
                key = view
            content_accuracy = results['content_accuracies'][key]
            shape_decisions = results['shape_decisions'][key]
            texture_decisions = results['texture_decisions'][key]
            all_decisions = results['total_decisions'][key]

            shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion, shape_bias = \
                get_decision_proportion_and_shape_bias(shape_decisions, texture_decisions, all_decisions)
            shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se = (
                get_standard_errors(shape_decisions, texture_decisions, all_decisions))

            print(f"Shape bias of {view} (ID): {shape_bias}")
            # plot the decision proportions and shape bias
            # set up bar width and positions
            bar_width = 0.2

            # set a flattened index
            ind_flattened = ind_background * len(view) + ind_view

            # Plot bars for texture/shape decision proportions on primary y-axis
            ax1.bar(ind_flattened - 0.5 * bar_width * (1 / width_factor_decisions), shape_decision_proportion,
                    yerr=shape_decisions_se,
                    width=bar_width,
                    label='Texture Decision Proportion',
                    color=palette_shape_bias['shape_proportion'],
                    zorder=1)
            ax1.bar(ind_flattened + 0.5 * bar_width * (1 / width_factor_decisions), texture_decision_proportion,
                    yerr=texture_decisions_se,
                    width=bar_width,
                    label='Shape Decision Proportion',
                    color=palette_shape_bias['texture_proportion'],
                    zorder=1)

            ax3.bar(ind_flattened, shape_bias, width=bar_width * (1/width_factor_shape_bias), yerr=shape_bias_se,
                    label='Shape Bias', color=palette_shape_bias['shape_bias'],
                    zorder=1)

    # ax1 and ax2 settings
    ax1.set_ylim([decision_proportion_min, decision_proportion_max])
    ax3.set_ylim([shape_bias_min, shape_bias_max])
    ax1.title.set_text('IDV')
    ax3.title.set_text('IDV')

    # plot shape bias on viewpoint o.o.d. dataset
    for ind_background, background in enumerate(backgrounds):
        for ind_view, view in enumerate(views):
            # get cue-conflict results for current model
            result_file_name = 'shape_bias.pkl'
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, result_file_name)
            with open(result_path, 'rb') as file:
                results = pickle.load(file)
            if suffix:
                key = '_'.join([ood_view, suffix])
            else:
                key = ood_view
            content_accuracy = results['content_accuracies'][key]
            shape_decisions = results['shape_decisions'][key]
            texture_decisions = results['texture_decisions'][key]
            all_decisions = results['total_decisions'][key]

            shape_and_texture_decision_proportion, shape_decision_proportion, texture_decision_proportion, shape_bias = \
                get_decision_proportion_and_shape_bias(shape_decisions, texture_decisions, all_decisions)
            shape_and_texture_decisions_se, shape_decisions_se, texture_decisions_se, shape_bias_se = (
                get_standard_errors(shape_decisions, texture_decisions, all_decisions))

            print(f"Shape bias of {view} (HOV): {shape_bias}")
            # plot the decision proportions and shape bias
            # set up bar width and positions
            bar_width = 0.2

            # set a flattened index
            ind_flattened = ind_background * len(view) + ind_view

            ax2.bar(ind_flattened - 0.5 * bar_width * (1 / width_factor_decisions), shape_decision_proportion,
                    yerr=shape_decisions_se,
                    width=bar_width,
                    label='Texture Decision Proportion',
                    color=palette_shape_bias['shape_proportion'],
                    zorder=1)
            ax2.bar(ind_flattened + 0.5 * bar_width * (1 / width_factor_decisions), texture_decision_proportion,
                    yerr=texture_decisions_se,
                    width=bar_width,
                    label='Shape Decision Proportion',
                    color=palette_shape_bias['texture_proportion'],
                    zorder=1)

            ax4.bar(ind_flattened, shape_bias, width=bar_width * (1 / width_factor_shape_bias), yerr=shape_bias_se,
                    label='Shape Bias', color=palette_shape_bias['shape_bias'], zorder=1)

    ax2.set_ylim([decision_proportion_min, decision_proportion_max])
    ax4.set_ylim([shape_bias_min, shape_bias_max])
    ax2.title.set_text('HOV')
    ax4.title.set_text('HOV')

    # set common labels
    fig.text(0.2, 0.95 , 'Shape and Texture Decision Proportions', va='center', rotation='horizontal', fontsize=12)
    fig.text(0.45, 0.5, 'Shape Bias', va='center', rotation='horizontal', fontsize=12)

    # set ticks for everyone
    ax1.set_xticks(np.arange(len(backgrounds) * len(views)))
    ax1.set_xticklabels([item.rsplit(' ', 1)[0] for item in view_plot_name])
    ax2.set_xticks(np.arange(len(backgrounds) * len(views)))
    ax2.set_xticklabels([item.rsplit(' ', 1)[0] for item in view_plot_name])
    ax3.set_xticks(np.arange(len(backgrounds) * len(views)))
    ax3.set_xticklabels([item.rsplit(' ', 1)[0] for item in view_plot_name])
    ax4.set_xticks(np.arange(len(backgrounds) * len(views)))
    ax4.set_xticklabels([item.rsplit(' ', 1)[0] for item in view_plot_name])

    # set the spines
    ax1.spines[['top', 'right']].set_visible(False)
    ax2.spines[['top', 'right']].set_visible(False)
    ax3.spines[['top', 'right']].set_visible(False)
    ax4.spines[['top', 'right']].set_visible(False)

    # # set the grids
    # ax1.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax2.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax3.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    # ax4.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

    # Add horizontal line at 1/32 to show chance level
    ax1.axhline(y=1/32, color='gray', linestyle='--', linewidth=1, zorder=0, alpha=0.7)
    ax2.axhline(y=1/32, color='gray', linestyle='--', linewidth=1, zorder=0, alpha=0.7)
    ax3.axhline(y=1/2, color='gray', linestyle='--', linewidth=1, zorder=0, alpha=0.7)
    ax4.axhline(y=1/2, color='gray', linestyle='--', linewidth=1, zorder=0, alpha=0.7)

    # create color legend
    color_legend_left = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                    for color in [palette_shape_bias['texture_proportion'], palette_shape_bias['shape_proportion']]]
    color_legend_right = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8)
                    for color in [palette_shape_bias['shape_bias']]]
    ax2.legend(color_legend_left, ['Texture Decision Proportion', 'Shape Decision Proportion'], ncol=2,)
    ax4.legend(color_legend_right, ['Shape Bias'])

    # save the plot
    save_path = os.path.join(plot_dir, f'{model_name}_shape_bias.svg')
    plt.savefig(save_path, format='svg')
    plt.close(fig)


def get_shape_bias_by_category(shape_decisions, texture_decisions, all_decisions):
    shape_bias_by_category = {}
    for category in category_labels:
        total_decisions = shape_decisions.get(category, 0) + texture_decisions.get(category, 0)
        if total_decisions > 0:
            shape_bias = (np.sqrt(shape_decisions[category] / total_decisions) *
                          np.sqrt(shape_decisions[category] / all_decisions[category]))
        else:
            shape_bias = 0
        shape_bias_by_category[category] = shape_bias
    return shape_bias_by_category


def plot_shape_bias_comparison():
    # Collect shape bias data for matched, non-matched, and OOD datasets
    shape_bias_data_matched = []
    shape_bias_data_non_matched = []
    shape_bias_data_ood = []

    for background in backgrounds:
        for view in views:
            result_file_name = 'shape_bias.pkl'
            log_folder = '_'.join([background, view, res])
            result_path = os.path.join(log_dir, log_folder, result_file_name)

            with open(result_path, 'rb') as file:
                results = pickle.load(file)

            shape_decisions = results['shape_decisions'][view]
            texture_decisions = results['texture_decisions'][view]
            all_decisions = results['total_decisions'][view]

            shape_bias_by_category = get_shape_bias_by_category(shape_decisions, texture_decisions, all_decisions)

            for category, bias in shape_bias_by_category.items():
                shape_bias_data_matched.append(
                    {'Category': category, 'View': view, 'Shape Bias': bias, 'Dataset': 'Matched'})

            # Collect data for non-matched views
            other_views = [item for item in views if item != view]
            for other_view in other_views:
                shape_decisions = results['shape_decisions'][other_view]
                texture_decisions = results['texture_decisions'][other_view]
                all_decisions = results['total_decisions'][other_view]

                shape_bias_by_category = get_shape_bias_by_category(shape_decisions, texture_decisions, all_decisions)

                for category, bias in shape_bias_by_category.items():
                    shape_bias_data_non_matched.append(
                        {'Category': category, 'View': view, 'Shape Bias': bias, 'Dataset': 'Non-Matched'})

            # Collect data for OOD views
            ood_view = config['log']['eval']['ood_view']
            shape_decisions = results['shape_decisions'][ood_view]
            texture_decisions = results['texture_decisions'][ood_view]
            all_decisions = results['total_decisions'][ood_view]

            shape_bias_by_category = get_shape_bias_by_category(shape_decisions, texture_decisions, all_decisions)

            for category, bias in shape_bias_by_category.items():
                shape_bias_data_ood.append(
                    {'Category': category, 'View': view, 'Shape Bias': bias, 'Dataset': 'OOD'})

    # Convert to DataFrame
    df_shape_bias = pd.DataFrame(shape_bias_data_matched + shape_bias_data_non_matched + shape_bias_data_ood)
    df_shape_bias['View'].value_counts()

    # Plot shape bias by dataset type

    for dataset_type in ['Matched', 'Non-Matched', 'OOD']:
        plt.figure(figsize=(14, 20))
        ax = sns.barplot(data=df_shape_bias[df_shape_bias['Dataset'] == dataset_type], x='View', y='Shape Bias',
                         palette=palette, ci=None)

        # Add line plot to show trends of shape bias change for each category
        sns.lineplot(data=df_shape_bias[df_shape_bias['Dataset'] == dataset_type], x='View', y='Shape Bias',
                     hue='Category',
                     marker='o', ax=ax, color='grey', alpha=0.3, legend=False)

        # Statistical annotations
        # pairs = [(views[i], views[j]) for i in range(len(views))
        #          for j in range(i + 1, len(views))]
        # annotator = Annotator(ax, pairs, data=df_shape_bias[df_shape_bias['Dataset'] == dataset_type], x='View',
        #                       y='Shape Bias')
        # annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni')
        # annotator.apply_and_annotate()

        # Customize plot
        plt.ylim(0, 1)
        plt.title(f'Shape Bias by View and Category ({dataset_type} Dataset)')
        plt.xlabel('View')
        plt.ylabel('Shape Bias')

        # Save plot
        save_path = os.path.join(plot_dir, f'shape_bias_comparison_{dataset_type.lower()}.png')
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    for suffix in suffixes:
        plot_shape_bias(suffix)
        print('Done')
