import argparse
import os
import pickle

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot robustness')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config
    log_dir = args.log_dir
    plot_dir = args.plot_dir

    viewpoint = config['data']['viewpoint']
    distortion_types = config['distortion_types']
    model_folders = config['model']['model_folders']

    epochs = range(30, 31)
    severities = range(1, 6)

    # Define the number of rows and columns based on the number of distortion types
    num_distortions = len(distortion_types)
    num_rows = (num_distortions + 3) // 4  # Adjust this based on how many columns you want
    num_cols = min(4, num_distortions)  # Max 4 columns for a cleaner look

    for epoch in epochs:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))  # Adjust size as needed
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Iterate over distortion types and plot for each one
        for idx, distortion_type in enumerate(distortion_types):
            ax = axes[idx]  # Select the current subplot axis

            # For each model folder, plot the relative accuracy
            for model_folder in model_folders:
                log_model_epoch_folder = os.path.join(log_dir, model_folder, str(epoch), viewpoint)
                content_acc_path = os.path.join(log_model_epoch_folder, 'content_acc.pkl')
                with open(content_acc_path, 'rb') as f:
                    content_acc = pickle.load(f)

                distortion_acc_path = os.path.join(log_model_epoch_folder, 'distortion_acc.pkl')
                with open(distortion_acc_path, 'rb') as f:
                    distortion_acc = pickle.load(f)

                # Retrieve content accuracy for the epoch
                content_acc_epoch = content_acc[str(epoch)]

                # Prepare to plot results for the current distortion type
                relative_accuracies = []
                for severity in severities:
                    # Get distortion accuracy for each severity level
                    distortion_acc_value = distortion_acc[str(epoch)][distortion_type][str(severity)]

                    # Compute relative accuracy (distortion accuracy relative to content accuracy)
                    relative_acc = distortion_acc_value / content_acc_epoch
                    relative_accuracies.append(relative_acc)

                # Plot the relative accuracy for the current model folder on the same axis
                ax.plot(severities, relative_accuracies, label=model_folder)

            # Customize the current subplot
            ax.set_title(distortion_type)
            ax.set_xlabel('Noise Strength')
            ax.set_ylabel('Accuracy (Relative to Content)')
            ax.grid(True)
            ax.legend()

        # Remove any unused subplots (if distortion types < num_rows * num_cols)
        for i in range(num_distortions, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined plot
        plot_save_folder = os.path.join(log_dir, 'robustness')
        os.makedirs(plot_save_folder, exist_ok=True)
        plot_save_path = os.path.join(plot_save_folder, f'relative_accuracy_epoch_{epoch}_{viewpoint}.png')
        plt.savefig(plot_save_path)

        # Show the plot (if running in an interactive environment)
        # plt.show()  # Uncomment this if you want to see the plot in an interactive environment

        # Clear the figure for the next plot
        plt.clf()
