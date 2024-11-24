import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler


def get_model(model_name, pretrained=True, **kwargs):
    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        transforms = models.AlexNet_Weights.DEFAULT.transforms
        model = models.alexnet(weights=weights, **kwargs)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, kwargs.get('num_classes', 32))
    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        transforms = models.ResNet18_Weights.DEFAULT.transforms
        model = models.resnet18(weights=weights, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, kwargs.get('num_classes', 32))
    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        transforms = models.ResNet34_Weights.DEFAULT.transforms
        model = models.resnet34(weights=weights, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, kwargs.get('num_classes', 32))
    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        transforms = models.ResNet50_Weights.DEFAULT.transforms
        model = models.resnet50(weights=weights, **kwargs)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, kwargs.get('num_classes', 32))
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model, transforms


def get_optimizer(optimizer_name, parameters, lr, **kwargs):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, momentum=kwargs.get('momentum', 0.9),
                              weight_decay=kwargs.get('weight_decay', 0.001))
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
    return optimizer


def get_criterion(criterion_name):
    if criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif criterion_name == 'L1Loss':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Criterion {criterion_name} is not supported.")
    return criterion


def get_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 30),
                                        gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=kwargs.get('mode', 'min'),
                                                   factor=kwargs.get('factor', 0.1),
                                                   patience=kwargs.get('patience', 10))
    else:
        raise ValueError(f"Scheduler {scheduler_name} is not supported.")
    return scheduler


def get_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)
