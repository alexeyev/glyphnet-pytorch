# coding: utf-8
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset import HogweedClassificationDataset
from plant_clef_resnet import load_plant_clef_resnet18


def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
          loss_function: nn.Module, current_epoch_number: int = 0,
          device: torch.device = None, batch_reports_interval: int = 10):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        # throwing away the gradients
        optimizer.zero_grad()

        # predicting scores
        output = model(data.to(device))

        # computing the error
        loss = loss_function(output, target.unsqueeze(dim=-1).float().to(device))

        # saving loss for stats
        loss_accum += loss.item() / len(data)

        # computing gradients
        loss.backward()

        # updating the model's weights
        optimizer.step()

        if batch_idx % batch_reports_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAveraged Epoch Loss: {:.6f}'.format(
                current_epoch_number,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_accum / (batch_idx + 1)))


def sigmoid2predictions(output: torch.Tensor) -> torch.Tensor:
    """ model.predict(X) based on sigmoid scores """
    return (torch.sign(output - 0.5) + 1) / 2


def test(model, test_loader, loss_function, device):
    """ Testing an already trained model using the provided data from `test_loader` """

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            target = target.float().unsqueeze(dim=-1).to(device)
            output = model(data.to(device))
            pred = sigmoid2predictions(output)
            test_loss += loss_function(output, target).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('...validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def set_parameter_requires_grad(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


if __name__ == "__main__":

    from argparse import ArgumentParser
    from datetime import datetime

    parser = ArgumentParser()
    parser.add_argument("--seed", default=160)
    parser.add_argument("--val_fraction", default=0.4)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--l1size", default=128)
    parser.add_argument("--dropout", default=0.8)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--unfreeze", default=True)
    parser.add_argument("--epochs_unfreeze", default=50)
    args = parser.parse_args()

    train_set = HogweedClassificationDataset(root="prepared_data/images_train_resized",
                                             transform=transforms.Compose([transforms.ToTensor()]))
