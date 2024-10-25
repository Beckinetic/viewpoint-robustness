import torch

from src.model.models import get_device

device = get_device()


def test_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # No need to track gradients during testing
        for data, targets in test_dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            test_loss += loss.item() * data.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / total
    accuracy = correct / total * 100  # As a percentage

    return avg_test_loss, accuracy
