import argparse
import os
import pickle
import warnings

import yaml
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss and accuracy')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--plot-dir', type=str, default='plots/', help='Directory to save logs')
    return parser.parse_args()


def plot_view_loss():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    for background in config['log']['background']:
        for view in config['log']['view']:
            for res in config['log']['res']:
                log_folder = '_'.join([background, view, res])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))

                other_views = [other_view for other_view in config['log']['view'] if other_view != view]
                other_log_folders = []
                view_labels = {'f': 'free',
                               'r': 'restricted',
                               'fx': 'fixed'}
                for other_view in other_views:
                    other_log_folders.append('_'.join([background, other_view, res]))

                other_result_paths = []
                for other_log_folder in other_log_folders:
                    other_result_paths.append(os.path.join(args.log_dir, other_log_folder,
                                                           '_'.join([model_name, 'log', 'data.pkl'])))

                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                train_loss = result['tl']
                val_loss = result['vl']

                epochs = range(1, len(train_loss) + 1)
                fig, ax1 = plt.subplots()
                # fig.set_size_inches(8, 4.5)
                fig.tight_layout()

                ax1.plot(epochs, train_loss, label=f'TL: {view_labels[view]} view', linestyle='-')
                ax1.plot(epochs, val_loss, label=f'VL: {view_labels[view]} view', linestyle='--')

                for ind, other_view in enumerate(other_views):
                    other_result_path = other_result_paths[ind]

                    with open(other_result_path, 'rb') as f:
                        other_result = pickle.load(f)

                    other_train_loss = other_result['tl']
                    other_val_loss = other_result['vl']
                    ax1.plot(epochs, other_train_loss, label=f'TL: {view_labels[other_view]} view', linestyle='-')
                    ax1.plot(epochs, other_val_loss, label=f'VL: {view_labels[other_view]} view', linestyle='--')


                # overall plotting
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_ylim([config['plot']['loss_min'], config['plot']['loss_max']])

                ax1.legend(loc='upper right')
                plt.tight_layout()
                title = '_'.join([model_name, log_folder])
                # plt.title(title)
                print(f"{title}: The min train loss is {min(train_loss)}, min validation loss is {min(val_loss)}")

                os.makedirs(os.path.join(args.plot_dir, log_folder), exist_ok=True)
                save_path = os.path.join(args.plot_dir, log_folder, '_'.join([title, 'view_comparison_loss.png']))
                plt.savefig(save_path)


def plot_view_acc():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    for background in config['log']['background']:
        for view in config['log']['view']:
            for res in config['log']['res']:
                log_folder = '_'.join([background, view, res])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))

                other_views = [other_view for other_view in config['log']['view'] if other_view != view]
                other_log_folders = []
                view_labels = {'f': 'free',
                               'r': 'restricted',
                               'fx': 'fixed'}
                for other_view in other_views:
                    other_log_folders.append('_'.join([background, other_view, res]))

                other_result_paths = []
                for other_log_folder in other_log_folders:
                    other_result_paths.append(os.path.join(args.log_dir, other_log_folder,
                                                           '_'.join([model_name, 'log', 'data.pkl'])))

                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                train_accs = result['ta']
                val_accs = result['va']

                epochs = range(1, len(train_accs) + 1)
                fig, ax1 = plt.subplots()
                # fig.set_size_inches(8, 4.5)
                fig.tight_layout()

                ax1.plot(epochs, train_accs, label=f'TA: {view_labels[view]} view', linestyle='-')
                ax1.plot(epochs, val_accs, label=f'VA: {view_labels[view]} view', linestyle='--')

                for ind, other_view in enumerate(other_views):
                    other_result_path = other_result_paths[ind]

                    with open(other_result_path, 'rb') as f:
                        other_result = pickle.load(f)

                    other_train_accs = other_result['ta']
                    other_val_accs = other_result['va']
                    ax1.plot(epochs, other_train_accs, label=f'TA: {view_labels[other_view]} view', linestyle='-')
                    ax1.plot(epochs, other_val_accs, label=f'VA: {view_labels[other_view]} view', linestyle='--')

                # overall plotting
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim([config['plot']['acc_min'], config['plot']['acc_max']])

                ax1.legend(loc='lower right')
                plt.tight_layout()
                title = '_'.join([model_name, log_folder])
                # plt.title(title)
                print(f"{title}: The max train acc is {max(train_accs)}, max validation acc is {max(val_accs)}")

                os.makedirs(os.path.join(args.plot_dir, log_folder), exist_ok=True)
                save_path = os.path.join(args.plot_dir, log_folder, '_'.join([title, 'view_comparison_acc.png']))
                plt.savefig(save_path)


def plot_loss_acc():
    args = parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    for background in config['log']['background']:
        for view in config['log']['view']:
            for res in config['log']['res']:
                log_folder = '_'.join([background, view, res])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                train_losses = result['tl']
                train_accs = result['ta']
                val_losses = result['vl']
                val_accs = result['va']

                epochs = range(1, len(train_losses) + 1)
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()

                ax1.plot(epochs, train_losses, label='Train loss', color='blue')
                ax1.plot(epochs, val_losses, label='Val loss', color='red')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_ylim([config['plot']['loss_min'], config['plot']['loss_max']])

                ax2.plot(epochs, train_accs, label='Train acc', color='blue', linestyle='-')
                ax2.plot(epochs, val_accs, label='Val acc',color='red', linestyle='--')
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


def plot_multival_loss_acc():
    args = parse_args()
    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['model_name']
    for background in config['log']['background']:
        for view in config['log']['view']:
            for res in config['log']['res']:
                log_folder = '_'.join([background, view, res])
                result_path = os.path.join(args.log_dir, log_folder, '_'.join([model_name, 'log', 'data.pkl']))
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)

                train_losses = result['tl']
                train_accs = result['ta']
                val_losses = result['vl']
                val_accs = result['va']

                epochs = range(1, len(train_losses) + 1)
                all_views = ['f', 'r', 'fx']
                view_colors = {
                    'f': 'blue',
                    'r': 'red',
                    'fx': 'green',
                }
                other_views = [item for item in all_views if item != view]

                # edit from here
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(epochs, train_losses, label='Train loss', color='grey')
                ax1.plot(epochs, val_losses[0], label=f'Val loss {view}', color=view_colors[view])
                ax1.plot(epochs, val_losses[1], label=f'Val loss {other_views[0]}', color=view_colors[other_views[0]])
                ax1.plot(epochs, val_losses[2], label=f'Val loss {other_views[1]}', color=view_colors[other_views[1]])
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_ylim([config['plot']['loss_min'], config['plot']['loss_max']])

                ax2.plot(epochs, train_accs, label='Train acc', color='grey', linestyle='-')
                ax2.plot(epochs, val_accs[0], label=f'Val acc {view}', color=view_colors[view], linestyle='dashed')
                ax2.plot(epochs, val_accs[1], label=f'Val acc {other_views[0]}', color=view_colors[other_views[0]], linestyle='dashed')
                ax2.plot(epochs, val_accs[2], label=f'Val acc {other_views[1]}', color=view_colors[other_views[1]], linestyle='dashed')
                ax2.set_ylabel('Accuracy')
                ax2.set_ylim([config['plot']['acc_min'], config['plot']['acc_max']])

                ax1.legend(loc='lower right')
                ax2.legend(loc='upper right')
                plt.tight_layout()
                title = '_'.join([model_name, log_folder])
                plt.title(title)

                os.makedirs(os.path.join(args.plot_dir, log_folder), exist_ok=True)
                save_path = os.path.join(args.plot_dir, log_folder, '_'.join([title, 'multival_loss_acc.png']))
                plt.savefig(save_path)




if __name__ == '__main__':
    # plot_loss_acc()
    # plot_view_loss()
    # plot_view_acc()
    plot_multival_loss_acc()
    print('Done')
