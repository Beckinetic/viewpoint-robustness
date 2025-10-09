import logging
import os
import pickle

import yaml
import torch
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.create_dataset import create_dataset, train_val_split
from src.model.models import get_model, get_optimizer, get_criterion, get_scheduler, get_device

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('config', type=str, help='Path to the configuration file')
parser.add_argument('--data-dir', type=str, default='data/', help='Directory to fetch the data')
parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save models')
args = parser.parse_args()

data_dir = args.data_dir
log_dir = args.log_dir
model_dir = args.model_dir

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

train_batch_size = config['data']['train']['batch_size']
train_num_classes = config['data']['train']['num_classes']
train_views = config['data']['train']['view']
train_res = config['data']['train']['res']
train_backgrounds = config['data']['train']['background']

val_batch_size = config['data']['validation']['batch_size']
val_num_classes = config['data']['validation']['num_classes']
val_views = config['data']['validation']['view']
val_res = config['data']['validation']['res']
val_backgrounds = config['data']['validation']['background']

backbone = config['model']['backbone']
pretrained = config['model']['pretrained']

epochs = config['model']['epochs']
checkpoint = config['model']['checkpoint']

instance = config['model']['instance']
print(f"Now training instance: {instance}")

optimizer_name = config['optimizer']['optimizer_name']
learning_rate = config['optimizer']['learning_rate']
weight_decay = config['optimizer']['weight_decay']
momentum = config['optimizer']['momentum']

criterion_name = config['criterion']['criterion_name']

scheduler_name = config['scheduler']['scheduler_name']
step_size = config['scheduler']['step_size']
gamma = config['scheduler']['gamma']


def prepare_data(data_folder):
    data_path = os.path.join(data_dir, data_folder)

    # ImageNet data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = create_dataset(data_path, label_path=None, transform=transform)

    train_dataset, val_dataset = train_val_split(dataset)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    return train_dataset, val_dataset


def train(data_folder, train_dataset, val_datasets):
    # Prepare data
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloaders = {}
    for val_ind, key in enumerate(val_datasets):
        val_dataloader = DataLoader(val_datasets[key], batch_size=val_batch_size, shuffle=False)
        val_dataloaders[key] = val_dataloader
    tqdm.write(f'Train dataset size: {len(train_dataset)}')
    tqdm.write(f'Val dataset number: {len(val_datasets)}')

    # model
    model, _ = get_model(backbone, pretrained=pretrained, num_classes=train_num_classes)
    model.eval()

    # device
    device = get_device()
    tqdm.write(f'Device: {device}')
    model = model.to(device)

    # optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

    # criterion
    criterion = get_criterion(criterion_name)

    # scheduler
    scheduler = get_scheduler(scheduler_name, optimizer, step_size=step_size, gamma=gamma)

    # save untrained model as baseline
    if pretrained:
        is_pretrained = 'pretrained'
    else:
        is_pretrained = 'scratch'
    untrained_model_path = f"{model_dir}/{data_folder}/{backbone}_{is_pretrained}_instance{instance}_epoch_0.pth"
    os.makedirs(os.path.dirname(untrained_model_path), exist_ok=True)
    torch.save(model.state_dict(), untrained_model_path)

    # training Loop
    # initialise losses and accuracies
    train_losses = []
    train_accs = []
    val_losses = {}
    val_accs = {}
    for val_ind, key in enumerate(val_datasets):
        val_losses[key] = []
        val_accs[key] = []

    for epoch in tqdm(range(epochs), desc=f'Training {backbone} on {data_folder}'):
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
        for val_ind, key in enumerate(val_dataloaders):
            val_dataloader = val_dataloaders[key]
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

            val_losses[key].append(val_loss / len(val_dataloader.dataset))
            val_accs[key].append(100 * val_correct / total_val)

        # run scheduler
        scheduler.step()

        # save checkpoints
        if (epoch + 1) % checkpoint == 0:
            checkpoint_path = f"{model_dir}/{data_folder}/{backbone}_{is_pretrained}_instance{instance}_epoch_{epoch + 1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        # save logs
        log_path = f"{log_dir}/{data_folder}/loss_acc_log/{backbone}_{is_pretrained}_instance{instance}_log_epoch_{epoch + 1}.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Train Loss: {train_loss}\n")
            log_file.write(f"Epoch {epoch + 1}, Train Accuracy: {100 * train_correct / total}\n")
            for val_ind, key in enumerate(val_dataloaders):
                log_file.write(f"Epoch {epoch + 1}, Validation {val_ind} Loss: {val_losses[key][epoch]}\n")
                log_file.write(f"Epoch {epoch + 1}, Validation {val_ind} Accuracy: {val_accs[key][epoch]}\n")

        # print log
        tqdm.write(f"Epoch {epoch + 1}, Train Loss: {train_loss}\n")
        tqdm.write(f"Epoch {epoch + 1}, Train Accuracy: {100 * train_correct / total}\n")
        for val_ind, key in enumerate(val_dataloaders):
            tqdm.write(f"Epoch {epoch + 1}, Validation {val_ind} Loss: {val_losses[key][epoch]}\n")
            tqdm.write(f"Epoch {epoch + 1}, Validation {val_ind} Accuracy: {val_accs[key][epoch]}\n")

    # save congregated loss and acc data
    log_data_dict = {
        'tl': train_losses,
        'vl': val_losses,
        'ta': train_accs,
        'va': val_accs
    }
    log_data_dict_path = f"{log_dir}/{data_folder}/{backbone}_{is_pretrained}_instance{instance}_log_data.pkl"
    with open(log_data_dict_path, 'wb') as f:
        pickle.dump(log_data_dict, f)


def main():
    # combine the same datasets of the same view
    combined_train_datasets = {}
    for view in train_views:
        train_datasets = []
        train_data_folders = []

        for background in train_backgrounds:
            for res in train_res:
                train_data_folder = '_'.join([background, view, res])
                train_data_folders.append(train_data_folder)

        for train_data_folder in train_data_folders:
            train_dataset, _ = prepare_data(train_data_folder)
            train_datasets.append(train_dataset)

        combined_train_dataset = ConcatDataset(train_datasets)
        combined_train_datasets[view] = combined_train_dataset

    combined_val_datasets = {}
    for view in val_views:
        val_datasets = []
        val_data_folders = []

        for background in val_backgrounds:
            for res in val_res:
                val_data_folder = '_'.join([background, view, res])
                val_data_folders.append(val_data_folder)

        for val_data_folder in val_data_folders:
            _, val_dataset = prepare_data(val_data_folder)
            val_datasets.append(val_dataset)

        combined_val_dataset = ConcatDataset(val_datasets)
        combined_val_datasets[view] = combined_val_dataset

    for ind_view, view in enumerate(train_views):
        data_folder = '_'.join(['combined', view, 'combined'])
        combined_train_dataset = combined_train_datasets[view]
        print(f'Combined dataset name: {data_folder}')
        print(f'Combined train dataset size: {len(combined_train_dataset)}')
        for val_ind, key in enumerate(combined_val_datasets):
            print(f'Combined validation dataset {val_ind} size: {len(combined_val_datasets[key])}')

        train(data_folder, combined_train_dataset, combined_val_datasets)


if __name__ == '__main__':
    main()
