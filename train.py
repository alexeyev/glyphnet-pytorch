# coding: utf-8

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from dataset import GlyphData
from model import Glyphnet


def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
          loss_function: nn.Module, current_epoch_number: int = 0,
          device: torch.device = None, batch_reports_interval: int = 100):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        # throwing away the gradients
        optimizer.zero_grad()

        # predicting scores
        output = model(data.to(device))

        # computing the error
        loss = loss_function(output, target.to(device))

        # saving loss for stats
        loss_accum += loss.item() / len(data)

        # computing gradients
        loss.backward()

        # updating the model's weights
        optimizer.step()

        if batch_idx % batch_reports_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAveraged Epoch Loss: {:.6f}'.format(
                current_epoch_number + 1,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_accum / (batch_idx + 1)))


def softmax2predictions(output: torch.Tensor) -> torch.Tensor:
    """ model.predict(X) based on softmax scores """
    return torch.topk(output, k=1, dim=-1).indices.flatten()


def test(model: nn.Module, test_loader: DataLoader, loss_function: nn.Module, device):
    """ Testing an already trained model using the provided data from `test_loader` """

    model.eval()
    test_loss, correct = 0, 0
    all_predictions, all_gold = [], []

    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output = model(data.to(device))
            pred = softmax2predictions(output)
            test_loss += loss_function(output, target).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_predictions.append(pred.numpy())
            all_gold.append(target.numpy())

    test_loss /= len(test_loader.dataset)

    print('...Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)

    print("Acc.: %2.2f%%; F-macro: %2.2f%%\n" % (accuracy_score(y_true, y_pred) * 100,
                                           f1_score(y_true, y_pred, average="macro") * 100))


if __name__ == "__main__":

    from argparse import ArgumentParser
    from os import listdir

    parser = ArgumentParser()
    parser.add_argument("--seed", default=160)
    parser.add_argument("--val_fraction", default=0.3)
    parser.add_argument("--batch_size", default=8)
    # parser.add_argument("--l1size", default=128)
    # parser.add_argument("--dropout", default=0.8)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--train_path", default="prepared_data/train/")
    parser.add_argument("--test_path", default="prepared_data/test/")
    args = parser.parse_args()

    train_labels = {l:i for i, l in enumerate(sorted([p.strip("/") for p in listdir(args.train_path)]))}

    train_set = GlyphData(root=args.train_path, class_to_idx=train_labels,
                          transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                        transforms.ToTensor()]))

    print("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)),
        train_set.targets,
        # stratify=train_set.targets,
        test_size=args.val_fraction,
        shuffle=True,
        random_state=args.seed
    )

    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices),
                                             shuffle=False,
                                             batch_size=128)

    print("CUDA available?", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up a model...")

    model = Glyphnet().to(device)
    optimizer = optim.AdamW(model.parameters(), amsgrad=True)
    loss_function = loss.CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, loss_function, epoch, device)
        test(model, val_loader, loss_function, device)

        print("Goodness of fit (evaluation on train):")
        test(model, train_loader, loss_function, device)

    #  FINAL EVALUATION

    test_labels_set = {l for l in [p.strip("/") for p in listdir(args.test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(root=args.test_path, class_to_idx=test_labels,
                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                       transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    model.eval()

    print("Checking quality on test set:")
    test(model, test_loader, loss_function, device)
