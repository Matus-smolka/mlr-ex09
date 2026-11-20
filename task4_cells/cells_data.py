"""The density target, the Dataset and the losses that Task 4's network is trained on.

Shared by train_cells.py (which chooses the arm), probe_cells.py (which re-runs the input variants to
measure how much of a count comes from the total brightness) and any later check, so that "the same
arm" means the same code in all three rather than three copies that drift apart.

The pieces, and why each is shaped as it is:

  density()      a Gaussian on every annotated centre, so sum(map) == count. scipy's gaussian_filter
                 reflects at the border, which folds the truncated tail back in and keeps that exact;
                 mode="constant" (what a zero-padded convolution layer does) loses up to 2-4 % on the
                 images with many border centres - measured in classical_cells.py section 6.
  CellsDataset   one image -> (1, 256, 256) input and a (256, 256) target. Flips and 90-degree turns
                 move the target along with the image; brightness jitter changes only the image, which
                 is the point - turning the lamp up does not add cells; input="zscore" standardizes
                 per image, which leaves the sum of the pixels at zero and so destroys the count
                 information carried by total brightness. gain/offset exist for the probe, which
                 measures what a trained network does when the lamp moves.
  make_loss()    'mse', 'l1', or 'mse+lam' which adds the squared error of the SUM of the map, written
                 per pixel so that it is in the same units as the MSE and lambda says how many times
                 more than one pixel the total is worth.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import Dataset


def density(dots, sigma, cache={}):
    """Every image's density target at once, cached per sigma."""
    key = (id(dots), sigma)
    if key not in cache:
        cache[key] = np.stack([ndimage.gaussian_filter(v.astype(np.float64), sigma)
                               for v in dots]).astype(np.float32)
    return cache[key]


class CellsDataset(Dataset):
    """One image -> (1, 256, 256) input and a (256, 256) density target. Augmentation lives here.

    `augment` flips, turns and (for a raw input) rescales the image while the target stays the count
    it always was. `input="zscore"` standardizes per image after that, so a global gain or offset
    cannot be seen at all - which is exactly what it is there to test.
    """

    def __init__(self, images, maps, idx, input="raw", augment=False, gain=1.0, offset=0.0):
        self.x, self.m = images[idx], maps[idx]
        self.input, self.augment, self.gain, self.offset = input, augment, gain, offset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) * self.gain + self.offset
        m = self.m[i].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5:
                img, m = img[::-1], m[::-1]
            k = int(np.random.randint(4))
            img, m = np.rot90(img, k), np.rot90(m, k)
            if self.input == "raw":                       # else normalization would remove it anyway
                img = img * np.random.uniform(0.7, 1.4) + np.random.uniform(-0.05, 0.05)
        if self.input == "zscore":
            img = (img - img.mean()) / (img.std() + 1e-6)
        return (torch.from_numpy(np.ascontiguousarray(img)[None]),
                torch.from_numpy(np.ascontiguousarray(m)))


def make_loss(kind):
    """'mse', 'l1', or 'mse+<lam>': the last adds the squared error of the sum of the map."""
    if kind == "mse":
        return lambda o, y: ((o[:, 0] - y) ** 2).mean()
    if kind == "l1":
        return lambda o, y: (o[:, 0] - y).abs().mean()
    if kind.startswith("mse+"):
        lam = float(kind[4:])
        return lambda o, y: ((o[:, 0] - y) ** 2).mean() \
            + lam * ((o[:, 0] - y).mean(dim=(-1, -2)) ** 2).mean()
    raise ValueError(kind)


def heads(name):
    """The last layer: nothing (a linear 1x1, free to go negative), softplus, or relu."""
    if name == "none":
        return nn.Identity()
    if name == "softplus":
        return nn.Softplus()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(name)


def count_of(maps):
    """The readout the task asks for: the count is the sum of the map."""
    return np.asarray(maps).reshape(len(maps), -1).sum(1)
