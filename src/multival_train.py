import os
import pickle

import yaml
import torch
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from tqdm import tqdm

from data.create_dataset import create_dataset, train_val_test_split
from model.models import get_model, get_optimizer, get_criterion, get_scheduler, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save models')
    return parser.parse_args()


def prepare_data(data_folder):
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    data_path = os.path.join(config['data']['root'], data_folder)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = create_dataset(data_path, label_path=None, transform=transform)

    train_dataset, val_dataset = train_val_test_split(dataset)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    return train_dataset, val_dataset


def multival_train(data_folder, train_dataset, val_datasets):
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # prepare data
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_dataloaders = []
    for val_dataset in val_datasets:
        val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=True)
        print(f'Validation dataset size: {len(val_dataloader)}')
        val_dataloaders.append(val_dataloader)

    # model
    model, _ = get_model(config['model']['model_name'],
                         pretrained=config['model']['pretrained'],
                         num_classes=config['data']['num_classes'])
    model.eval()

    # device
    device = get_device()
    print(f'Device: {device}')
    model = model.to(device)

    # optimizer
    optimizer = get_optimizer(config['optimizer']['optimizer_name'],
                              model.parameters(),
                              config['optimizer']['learning_rate'],
                              momentum=config['optimizer']['momentum'],
                              weight_decay=config['optimizer']['weight_decay'])

    # criterion
    criterion = get_criterion(config['criterion']['criterion_name'])

    # scheduler
    scheduler = get_scheduler(config['scheduler']['scheduler_name'],
                              optimizer,
                              step_size=config['scheduler']['step_size'],
                              gamma=config['scheduler']['gamma'])

    # save untrained model as baseline
    baseline_path = f"{args.model_dir}/{data_folder}/{config['model']['model_name']}_epoch_0.pth"
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    torch.save(model.state_dict(), baseline_path)

    # training Loop
    # initialise losses and accuracies
    train_losses = []
    train_accs = []
    val_losses = {}
    val_accs = {}
    for val_ind, val_dataset in enumerate(val_datasets):
        val_losses[val_ind] = []
        val_accs[val_ind] = []

    model_name = config['model']['model_name']
    for epoch in tqdm(range(config['training']['epochs']), desc=f'Training {model_name} on {data_folder}'):
        # train the model
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            # training step
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_losses.append(train_loss / len(train_dataloader.dataset))
        train_accs.append(100 * train_correct / total)

        # validate model
        model.eval()
        for val_ind, val_dataloader in enumerate(val_dataloaders):
            val_loss = 0.0
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_losses[val_ind].append(val_loss / len(val_dataloader.dataset))
            val_accs[val_ind].append(100 * val_correct / total_val)

        # run scheduler
        scheduler.step()

        # save checkpoints
        if (epoch + 1) % config['training'].get('check_point', 1) == 0:
            checkpoint_path = f"{args.model_dir}/{data_folder}/{config['model']['model_name']}_epoch_{epoch + 1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        # save logs
        log_path = f"{args.log_dir}/{data_folder}/{config['model']['model_name']}_log_epoch_{epoch + 1}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Train Loss: {train_loss}\n")
            log_file.write(f"Epoch {epoch + 1}, Train Accuracy: {100 * train_correct / total}\n")
            for val_ind, val_dataloader in enumerate(val_dataloaders):
                log_file.write(f"Epoch {epoch + 1}, Validation {val_ind} Loss: {val_losses[val_ind][epoch]}\n")
                log_file.write(f"Epoch {epoch + 1}, Validation {val_ind} Accuracy: {val_accs[val_ind][epoch]}\n")

    # save congregated loss and acc data
    log_data_dict = {
        'tl': train_losses,
        'vl': val_losses,
        'ta': train_accs,
        'va': val_accs
    }
    log_data_dict_path = f"{args.log_dir}/{data_folder}/{config['model']['model_name']}_log_data.pkl"
    with open(log_data_dict_path, 'wb') as file:
        pickle.dump(log_data_dict, file)


def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # combine the same datasets of the same view
    for view in config['data']['view']:
        data_folders = []
        for background in config['data']['background']:
            for res in config['data']['res']:
                data_folder = '_'.join([background, view, res])
                data_folders.append(data_folder)

        train_datasets = []
        val_datasets = []
        for data_folder in data_folders:
            train_dataset, val_dataset = prepare_data(data_folder)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)
        combined_val_datasets = [combined_val_dataset]

        # make other view validation sets
        all_views = ['f', 'r', 'fx']
        other_views = [item for item in all_views if item != view]
        for other_view in other_views:
            data_folders = []
            for background in config['data']['background']:
                for res in config['data']['res']:
                    data_folder = '_'.join([background, other_view, res])
                    data_folders.append(data_folder)

            val_datasets = []
            for data_folder in data_folders:
                _, val_dataset = prepare_data(data_folder)
                val_datasets.append(val_dataset)

            combined_val_dataset = ConcatDataset(val_datasets)
            combined_val_datasets.append(combined_val_dataset)

        data_folder = '_'.join(['combined', view, 'combined'])
        print(f'Combined dataset name: {data_folder}')
        print(f'Combined train dataset size: {len(combined_train_dataset)}')
        for val_ind, combined_val_dataset in enumerate(combined_val_datasets):
            print(f'Combined validation dataset {val_ind} size: {len(combined_val_dataset)}')

        multival_train(data_folder, combined_train_dataset, combined_val_datasets)


if __name__ == '__main__':
    main()
