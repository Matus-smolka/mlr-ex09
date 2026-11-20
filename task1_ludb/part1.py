"""Part 1 of ex09_student.ipynb, copied verbatim: conv_block, Encoder, UNet, train, predict.

Task 1 says to use `train`, `predict` and `UNet` of Part 1 without changing them, so this file is
that code with nothing added to it - kept in one place because both `train_ludb.py` and
`ablate_repair.py` need it, and a copy in each would be two copies to keep honest.

The only lines that are not from the notebook are the imports and the two globals `device` and
`num_workers` that the notebook's own `train` cell expects to exist.
"""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 0


def conv_block(dim, c_in, c_out):
    Conv = nn.Conv1d if dim == 1 else nn.Conv2d
    Norm = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
    return nn.Sequential(Conv(c_in, c_out, 3, padding=1), Norm(c_out), nn.ReLU(),
                         Conv(c_out, c_out, 3, padding=1), Norm(c_out), nn.ReLU())


class Encoder(nn.Module):
    def __init__(self, dim, c_in, widths):
        super().__init__()
        self.blocks = nn.ModuleList()
        for w in widths:
            self.blocks.append(conv_block(dim, c_in, w))
            c_in = w                                    # the next block takes this block's output
        self.pool = nn.MaxPool1d(2) if dim == 1 else nn.MaxPool2d(2)

    def forward(self, x):
        features = []                                   # the output of every block, finest first
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = self.pool(x)                        # halve the resolution between the blocks
            x = block(x)
            features.append(x)
        return features


class UNet(nn.Module):
    def __init__(self, dim, c_in, c_out, widths):
        super().__init__()
        self.encoder = Encoder(dim, c_in, widths)
        UpConv = nn.ConvTranspose1d if dim == 1 else nn.ConvTranspose2d
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.up = nn.ModuleList()
        self.decode = nn.ModuleList()
        for i in reversed(range(len(widths) - 1)):     # from the deepest level up, e.g. 64 -> 32 -> 16 -> 8
            self.up.append(UpConv(widths[i + 1], widths[i], 2, stride=2))
            self.decode.append(conv_block(dim, 2 * widths[i], widths[i]))
        self.head = Conv(widths[0], c_out, 1)          # 1 x 1 convolution

    def forward(self, x):
        features = self.encoder(x)                     # finest first
        x = features.pop()                             # the deepest features
        for up, decode in zip(self.up, self.decode):
            skip = features.pop()                      # the encoder's features one level finer
            x = up(x, output_size=skip.shape[2:])      # double the resolution (exactly the skip's size)
            x = decode(torch.cat([skip, x], dim=1))    # skip connection by concatenation
        return self.head(x)


def train(model, train_set, val_set, loss_fn, epochs, lr=1e-3, batch_size=16, num_workers=num_workers):
    """Adam on mini-batches; returns the model with the weights of its best validation epoch."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    train_curve, valid_curve = [], []
    best_loss = float("inf")
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_curve.append(np.mean(losses))

        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                losses.append(loss_fn(model(x), y).item())
        valid_curve.append(np.mean(losses))

        if valid_curve[-1] < best_loss:
            best_loss, best_epoch = valid_curve[-1], epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"{epochs} epochs in {time.perf_counter() - start:.0f} s; best validation loss {best_loss:.4f} in epoch {best_epoch}")
    plt.figure(figsize=(5, 2.5))
    plt.plot(range(1, epochs + 1), train_curve, label="training")
    plt.plot(range(1, epochs + 1), valid_curve, label="validation")
    plt.axvline(best_epoch, color="gray", linestyle=":")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    return model


def predict(model, dataset, batch_size=64):
    """All outputs of the model for a dataset (on the CPU), and the targets."""
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size):
            outputs.append(model(x.to(device)).cpu())
            targets.append(y)
    return torch.cat(outputs), torch.cat(targets)
