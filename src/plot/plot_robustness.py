import argparse
import os
import pickle

import yaml
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot robustness')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'rb') as f:
        config = yaml.safe_load(f)

    log_dir = args.log_dir
    plot_dir = args.plot_dir

    distortion_types = config['distortion_types']
    model_folders = config['model']['model_folders']

    views = config['data']['views']

    epochs = range(30, 31)
    severities = range(1, 6)
    suffixes = ['background', 'object', None]

    # Define the number of rows and columns based on the number of distortion types
    num_distortions = len(distortion_types)
    num_rows = 3
    num_cols = 7

    for view in views:
        for suffix in suffixes:
            if suffix is not None:
                result_folder = '_'.join(['ood', view, suffix])
            else:
                result_folder = '_'.join(['ood', view])

            for epoch in epochs:
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, num_rows * 5))  # Adjust size as needed
                axes = axes.flatten()  # Flatten the axes array for easy indexing

                for idx, distortion_type in enumerate(distortion_types):
                    ax = axes[idx]  # Select the current subplot axis

                    # for each model folder, plot the relative accuracy
                    for model_folder in model_folders:
                        log_model_epoch_folder = os.path.join(log_dir, model_folder, 'robustness', str(epoch),
                                                              result_folder)
                        content_acc_path = os.path.join(log_model_epoch_folder, 'content_acc.pkl')
                        with open(content_acc_path, 'rb') as f:
                            content_acc = pickle.load(f)

                        distortion_acc_path = os.path.join(log_model_epoch_folder, 'distortion_acc.pkl')
                        with open(distortion_acc_path, 'rb') as f:
                            distortion_acc = pickle.load(f)

                        # retrieve content accuracy for the epoch
                        content_acc_epoch = content_acc[str(epoch)]

                        # prepare to plot results for the current distortion type
                        accuracies = [content_acc_epoch]
                        severities_with_zero = [0] + list(severities)
                        for severity in severities:
                            # get distortion accuracy for each severity level
                            distortion_acc_value = distortion_acc[str(epoch)][distortion_type][str(severity)]

                            accuracies.append(distortion_acc_value)

                        # Plot the relative accuracy for the current model folder on the same axis
                        ax.plot(severities_with_zero, accuracies, label=model_folder)

                    # Customize the current subplot
                    ax.set_title(distortion_type)
                    ax.set_xlabel('Noise Strength')
                    ax.set_ylabel('Accuracy (Relative to Content)')
                    ax.set_ylim(0, 1)
                    ax.grid(True)
                    ax.legend()

                # Remove any unused subplots (if distortion types < num_rows * num_cols)
                for i in range(num_distortions, len(axes)):
                    fig.delaxes(axes[i])

                # Adjust layout to prevent overlap
                plt.tight_layout()

                # Save the combined plot
                plot_save_folder = os.path.join(plot_dir, 'robustness')
                os.makedirs(plot_save_folder, exist_ok=True)
                plot_save_path = os.path.join(plot_save_folder, f'acc_epoch_{epoch}_{result_folder}.png')
                plt.savefig(plot_save_path)

                # Show the plot (if running in an interactive environment)
                # plt.show()  # Uncomment this if you want to see the plot in an interactive environment

                # Clear the figure for the next plot
                plt.clf()


if __name__ == '__main__':
    main()
    print('Done')
