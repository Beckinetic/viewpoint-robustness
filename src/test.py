import argparse
import logging
import os
import pickle

import torch
import yaml
from torch.utils.data import ConcatDataset
from torchvision import transforms
from tqdm import tqdm

from src.model.test import test_model
from src.data.create_dataset import create_dataset, train_val_split
from src.model.models import get_model, get_device

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
    
batch_size = config['data']['test']['batch_size']
num_classes = config['data']['test']['num_classes']
test_views = config['data']['test']['view']
no_split = config['data']['test']['no_split']
test_res = config['data']['test']['res']
test_backgrounds = config['data']['test']['background']

backbone = config['model']['backbone']
pretrained = config['model']['pretrained']
instance = config['model']['instance']
print(f"Now testing instance: {instance}")

to_test_views = config['model']['to_test']['view']
to_test_res = config['model']['to_test']['res']
to_test_backgrounds = config['model']['to_test']['background']
max_epoch = config['model']['to_test']['max_epoch']

device = get_device()


def prepare_test_data(data_folder, if_split=True):
    data_path = os.path.join(data_dir, data_folder)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = create_dataset(data_path, label_path=None, transform=transform)

    if if_split:
        train_dataset, val_dataset = train_val_split(dataset)
        return val_dataset
    else:
        return dataset


def test():
    # load test set
    combined_test_datasets = {}
    for view in test_views:
        test_datasets = []
        test_data_folders = []

        for background in test_backgrounds:
            for res in test_res:
                test_data_folder = '_'.join([background, view, res])
                test_data_folders.append(test_data_folder)

        for test_data_folder in test_data_folders:
            if view in no_split:
                test_dataset = prepare_test_data(data_folder=test_data_folder, if_split=False)
            else:
                test_dataset = prepare_test_data(data_folder=test_data_folder, if_split=True)
            test_datasets.append(test_dataset)

        combined_test_dataset = ConcatDataset(test_datasets)
        combined_test_datasets[view] = combined_test_dataset

    for view in to_test_views:
        for res in to_test_res:
            for background in to_test_backgrounds:
                # find model folder and load result
                model_name = '_'.join([background, view, res])
                model_folder = os.path.join(model_dir, model_name)
                # save untrained model as baseline
                if pretrained:
                    is_pretrained = 'pretrained'
                else:
                    is_pretrained = 'scratch'
                result_path = os.path.join(log_dir, model_name, '_'.join([backbone, is_pretrained,
                                                                          'instance'+str(instance),
                                                                          'test', 'data.pkl']))

                # validation results and test results are stored together
                test_losses = {}
                test_accs = {}
                for ind_test, key in enumerate(combined_test_datasets):
                    test_losses[key] = []
                    test_accs[key] = []

                # load the model backbone
                general_backbone_name = backbone.split('_')[0]
                model, _ = get_model(general_backbone_name, pretrained=pretrained, num_classes=num_classes)
                model.to(device)

                for epoch in tqdm(range(max_epoch + 1), desc=f'Model {view}'):
                    model_path = os.path.join(model_folder, '_'.join([backbone, is_pretrained,
                                                                      'instance'+str(instance),
                                                                      'epoch', str(epoch) + '.pth']))
                    state_dict = torch.load(model_path, map_location=torch.device(device))
                    model.load_state_dict(state_dict)
                    model.eval()

                    # evaluate the model
                    for ind_test, key in enumerate(combined_test_datasets):
                        test_dataset = combined_test_datasets[key]
                        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                                      shuffle=False)
                        loss, accuracy = test_model(model, test_dataloader)
                        test_losses[key].append(loss)
                        test_accs[key].append(accuracy)

                # write the result
                with open(result_path, 'wb') as f:
                    pickle.dump({'tel': test_losses, 'tea': test_accs}, f)


if __name__ == '__main__':
    test()
    print('Done')
    