import json
import os.path
import random

from PIL import Image
from pathlib import Path
from io import BytesIO
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
from torchvision.models.vgg import VGG19_Weights

from src.model.models import get_device

# define device
device = get_device()

# freeze all VGG parameters since we're only optimizing the target image
vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)


def list_img_files(dir_):
    img_files = []

    for filename in os.listdir(dir_):
        if filename.endswith(".png"):
            img_files.append(os.path.join(dir_, filename))

    return img_files


def load_image(img_path, max_size=400, shape=None):
    """ Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims."""
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    image = (image * 255).astype(np.uint8)

    return image


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al. (2016)
    """

    # outputs and corresponding layers
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation output
                  '28': 'conv5_1'}

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    batch_size, d, h, w = tensor.size()

    tensor = tensor.view(d, h * w)  # can be just d,h*w

    gram = torch.mm(tensor, tensor.t())  # multiply with transpose

    # get the batch_size, depth, height, and width of the Tensor
    # reshape it, so we're multiplying the features for each channel
    # calculate the gram matrix

    return gram


def style_transfer(content, style, filename, output_dir, steps=2000):
    # get content and style features only once before forming the target image
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in
                   style_features}  # dic to store layer name : gram matrix

    # create a third "target" image and prep it for change
    # it is a good idea to start off with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    # for displaying the target image, intermittently
    show_every = 50

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)

    for ii in tqdm(range(1, steps + 1)):
        target_features = get_features(target, vgg)

        # content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # style loss
        style_loss = 0

        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]

            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            # get the "style" style representation
            style_gram = style_grams[layer]

            # the style loss for one layer, weighted
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        # update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii == steps:
            image = Image.fromarray(im_convert(target))
            image.save(os.path.join(output_dir, Path(filename).stem + '.jpg'))


def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def find_label_by_filename(data, filename):
    return next((item['label'] for item in data if item['filename'] == filename), None)


def find_filenames_by_label(data, label):
    return [item['filename'] for item in data if item['label'] == label]


def divide_labels(data, num):
    labels = list(set(item['label'] for item in data))
    random.shuffle(labels)
    return [labels[i::num] for i in range(num)]


# if __name__ == '__main__':
#     print("Run preview batch...")
#     print(f"Device is {device}")
#
#     content_dir = 'preview_batch/content'
#     style_dir = 'preview_batch/style'
#     output_dir = 'preview_batch/output'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     content_paths = list_img_files(content_dir)
#     styles_paths = list_img_files(style_dir)
#
#     for content_path in content_paths:
#         content_name = Path(content_path).stem
#         content = load_image(content_path).to(device)
#         for style_path in styles_paths:
#             style_name = Path(style_path).stem
#             style = load_image(style_path).to(device)
#             transferred_file_name = content_name + '_' +style_name + '.jpg'
#             print(f"Content: {content_name}; Style: {style_name}")
#             style_transfer(content, style, transferred_file_name, output_dir, steps=400)