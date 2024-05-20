import argparse
import os
import pickle

import yaml
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def main():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    for log_folder in config['log']['log_folders']:
        result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
        with open(result_path, 'rb') as f:
            result = pickle.load(f)

        train_losses = result['tl']
        train_accs = result['ta']
        val_losses = result['vl']
        val_accs = result['va']

        # # THIS IS A TEMPORARY THING AND SHOULD BE REMOVED LATER
        # train_losses = [train_loss for train_loss in train_losses]
        # val_losses = [val_loss for val_loss in val_losses]
        # print(train_losses, val_losses)

        epochs = range(1, len(train_losses) + 1)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(epochs, train_losses, label='Train loss')
        ax1.plot(epochs, val_losses, label='Val loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_ylim([config['plot']['loss_min'], config['plot']['loss_max']])

        ax2.plot(epochs, train_accs, label='Train acc')
        ax2.plot(epochs, val_accs, label='Val acc')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([config['plot']['acc_min'], config['plot']['acc_max']])

        ax1.legend(loc='lower right')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        title = '_'.join([model_name, log_folder])
        plt.title(title)

        os.makedirs(os.path.join(args.plot_dir, log_folder), exist_ok=True)
        save_path = os.path.join(args.plot_dir, log_folder, '_'.join([title, 'loss_acc.png']))
        plt.savefig(save_path)


if __name__ == '__main__':
    main()
    print('Done')
