import argparse
import os
import pickle

import yaml
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import ttest_rel

# Set seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = '500'  # medium weight
plt.rcParams['font.stretch'] = 'semi-expanded'  # slightly expanded
plt.rcParams['figure.dpi'] = 300

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--plot-dir', type=str, default='plots/')

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f.read())
log_dir = args.log_dir
plot_dir = args.plot_dir

views = config['log']['view']
backgrounds = config['log']['background']
res = config['log']['res']
eval_views = config['log']['eval']['view']
ood_views = config['log']['eval']['ood_view']
eval_types = config['log']['eval']['type']

palette = config['plot']['palette']
view_plot_name = config['plot']['view']
cam_types = config['plot']['cam_types']


def plot_border_cam():
    for eval_type in eval_types:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.subplots_adjust(left=0.1, right=0.85, wspace=0.3)

        for ind_cam, cam_type in enumerate(cam_types):
            data_matched = []
            data_non_matched = []
            data_ood = []
            cam_names = {
                'ic': 'Object',
                'bc': 'Edge',
                'ec': 'Background',
            }

            for ind_background, background in enumerate(backgrounds):
                for ind_view, view in enumerate(views):
                    eval_view = eval_views[ind_view]
                    log_folder = '_'.join([background, view, res])
                    border_cam_path = os.path.join(log_dir, log_folder, '.'.join([eval_type, 'pkl']))
                    with open(border_cam_path, 'rb') as f:
                        border_cam = pickle.load(f)

                    values_matched = border_cam[cam_type][eval_view].values()
                    data_matched.extend([(cam_type, views[ind_view], val) for i, val in enumerate(values_matched)])

                    # other_views = [item for item in eval_views if item != eval_view]
                    # for other_view in other_views:
                    #     values_non_matched = border_cam[cam_type][other_view].values()
                    #     data_non_matched.extend(
                    #         [(cam_type, views[ind_view], val) for i, val in enumerate(values_non_matched)])

                    for ood_view in ood_views:
                        values_ood = border_cam[cam_type][ood_view].values()
                        data_ood.extend([(cam_type, views[ind_view], val) for i, val in enumerate(values_ood)])

            # Convert data to DataFrame for seaborn
            df_matched = pd.DataFrame(data_matched, columns=['CAM Type', 'View', 'Value'])
            df_non_matched = pd.DataFrame(data_non_matched, columns=['CAM Type', 'View', 'Value'])
            df_ood = pd.DataFrame(data_ood, columns=['CAM Type', 'View', 'Value'])

            df_matched['Value'].fillna(df_matched['Value'].mean(), inplace=True)
            df_non_matched['Value'].fillna(df_non_matched['Value'].mean(), inplace=True)
            df_ood['Value'].fillna(df_ood['Value'].mean(), inplace=True)

            # Inside the plot_border_cam function
            pairs = [(views[i], views[j]) for i in range(len(views))
                     for j in range(i + 1, len(views))]
            print(pairs)

            # Plot matched data
            sns.barplot(ax=axes[0, ind_cam], x='View', y='Value', data=df_matched, palette=palette)
            axes[0, ind_cam].set_title(f'{cam_names[cam_type]} CAM - In-distribution')
            axes[0, ind_cam].set_ylim(0.5, 0.75)
            axes[0, ind_cam].set_xlabel("")
            axes[0, ind_cam].set_xticks([])
            # annotator = Annotator(axes[0, ind_cam], pairs, data=df_matched, x='View', y='Value')
            # annotator.configure(test='t-test_paired', text_format='star', comparisons_correction='bonferroni')
            # annotator.apply_and_annotate()

            # Plot non-matched data
            # sns.barplot(ax=axes[1, ind_cam], x='View', y='Value', data=df_non_matched, palette=palette)
            # axes[1, ind_cam].set_title(f'{cam_type.capitalize()} CAM - Non-Matched')
            # axes[1, ind_cam].set_ylim(0.5, 1)
            # annotator = Annotator(axes[1, ind_cam], pairs, data=df_non_matched, x='View', y='Value')
            # annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni')
            # annotator.apply_and_annotate()

            # Plot OOD data
            sns.barplot(ax=axes[1, ind_cam], x='View', y='Value', data=df_ood, palette=palette)
            axes[1, ind_cam].set_title(f'{cam_names[cam_type]} CAM - OOD')
            axes[1, ind_cam].set_ylim(0.5, 0.75)
            axes[1, ind_cam].set_xlabel("Models")
            axes[1, ind_cam].set_xticklabels(view_plot_name)
            # annotator = Annotator(axes[1, ind_cam], pairs, data=df_ood, x='View', y='Value')
            # annotator.configure(test='t-test_paired', text_format='star', comparisons_correction='bonferroni')
            # annotator.apply_and_annotate()
            if ind_cam == 2:
                axes[0, ind_cam].set_ylim(0.25, 0.5)
                axes[1, ind_cam].set_ylim(0.25, 0.5)
            if ind_cam > 0:
                axes[0, ind_cam].set_ylabel("")
                axes[1, ind_cam].set_ylabel("")
            else:
                axes[0, ind_cam].set_ylabel("CAM Value")
                axes[1, ind_cam].set_ylabel("CAM Value")

        # Add a legend for the bottom right plot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[1, len(cam_types) - 1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))

        # Save the plot
        save_path = os.path.join(plot_dir, '.'.join([eval_type, 'png']))
        plt.savefig(save_path)
        plt.close(fig)


if __name__ == '__main__':
    plot_border_cam()
    print('Done')
