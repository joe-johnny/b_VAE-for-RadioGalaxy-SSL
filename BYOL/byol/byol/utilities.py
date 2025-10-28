import logging
import numpy as np

import torch
import torch.utils.data as D
import matplotlib.pyplot as plt

from torch.optim import Optimizer
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from typing import Type, Any, Callable, Union, List, Optional
from torch.optim.lr_scheduler import LambdaLR
from math import cos, pi
from sklearn.model_selection import train_test_split

from byol.paths import Path_Handler


# Define paths
paths = Path_Handler()._dict()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def batch_eval(fn_dict, dset, batch_size=200):
    """
    Take functions which acts on data x,y and evaluates over the whole dataset in batches, returning a list of results for each calculated metric
    """
    n = len(dset)
    loader = DataLoader(dset, batch_size)

    # Fill the output dictionary with empty lists
    outs = {}
    for key in fn_dict.keys():
        outs[key] = []

    for x, y in loader:
        # Append result from each batch to list in outputs dictionary
        for key, fn in fn_dict.items():
            outs[key].append(fn(x, y))

    return outs


def CosineLinearWarmupScheduler(opt, warmup_epochs, max_epochs):
    """Cosine annealing with linear warmup.

    Args:
        opt (torch.optim.Optimizer): Optimizer to use.
        warmup_epochs (int): Number of epochs for warmup.
        total_epochs (int): Total number of epochs.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler.
    """
    # Reduce linear warmup epochs to account for 0th epoch
    warmup_epochs -= 1

    # Linear warmup schedule
    warmup_lr_schedule = lambda t: (t + 1) / warmup_epochs if t <= warmup_epochs else 1.0

    # Cosine annealing schedule
    cosine_lr_schedule = lambda t: 0.5 * (1 + cos(pi * t / max_epochs))

    # Combine schedules
    lr_schedule = lambda t: warmup_lr_schedule(t) * cosine_lr_schedule(t)

    return LambdaLR(opt, lr_schedule)


def _scheduler(
    opt: Optimizer,
    n_epochs: int,
    decay_type: str = "warmupcosine",
    warmup_epochs: int = 10,
):
    if decay_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
        return scheduler

    elif decay_type == "warmupcosine":
        scheduler = CosineLinearWarmupScheduler(
            opt,
            warmup_epochs,
            max_epochs=n_epochs,
        )
        return scheduler

    else:
        raise ValueError(decay_type)


def _optimizer(
    params,
    type: str = "sgd",
    lr: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 0.0000015,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    **kwargs,
):
    betas = (beta_1, beta_2)

    # for adam, lr is the step size and is modified by exp. moving av. of prev. gradients
    opts = {
        "adam": lambda p: torch.optim.Adam(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "adamw": lambda p: torch.optim.AdamW(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, weight_decay=weight_decay),
    }

    if type == "adam" and lr > 0.01:
        logging.warning(f"Learning rate {lr} may be too high for adam")

    opt = opts[type](params)

    return opt

    # Apply LARS wrapper if option is chosen
    # if config["optimizer"]["lars"]:
    #     opt = LARSWrapper(opt, eta=config["optimizer"]["trust_coef"])


def embed_dataset(encoder, data, batch_size=400):
    print("Embedding dataset...")
    train_loader = DataLoader(data, batch_size, shuffle=False)
    device = next(encoder.parameters()).device
    feature_bank = []
    target_bank = []
    for data in tqdm(train_loader):
        # Load data and move to correct device
        x, y = data

        x_enc = encoder(x.to(device))

        feature_bank.append(x_enc.squeeze().detach().cpu())
        target_bank.append(y.to(device).detach().cpu())

    # Save full feature bank for validation epoch
    feature_bank = torch.cat(feature_bank)
    target_bank = torch.cat(target_bank)

    return feature_bank, target_bank


# def log_examples(wandb_logger, dset, n=18):
#     save_list = []
#     count = 0
#     for x, _ in DataLoader(dset, 1):
#         if count > n:
#             break
#         x1, x2 = x
#         C, H, W = x1.shape[-3], x1.shape[-2], x1.shape[-1]

#         if C == 1:
#             fig = plt.figure(figsize=(13.0, 13.0))
#             grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0)

#             img_list = [x1, x2]
#             for ax, im in zip(grid, img_list):
#                 im = im.reshape((H, W, C))
#                 ax.axis("off")
#                 ax.imshow(im, cmap="hot")

#             plt.axis("off")
#             pil_img = fig2img(fig)
#             save_list.append(pil_img)
#             plt.close(fig)

#         else:
#             img_list = [x_i.view(C, H, W) for x_i in x]
#             img = make_grid(img_list)
#             # tens2pil = ToPILImage()
#             save_list.append(img)

#         count += 1

#     wandb_logger.log_image(key="image_pairs", images=save_list)

#--------------------
def log_examples(wandb_logger, dataset, model, n=18):
    """
    Logs n example pairs of BYOL views (x0, x1) generated by the model
    after VAE + augmentations.
    """
    save_list = []
    count = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for img, _ in loader:
        if count >= n:
            break

        img = img.to(model.device)   # move to GPU if needed
        # get x0, x1 from the model
        with torch.no_grad():
            x0, x1 = model.get_vae_views(img)

        # take first element (since batch=1)
        x0, x1 = x0[0].cpu(), x1[0].cpu()
        C, H, W = x0.shape

        if C == 1:
            fig = plt.figure(figsize=(6, 3))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

            for ax, im in zip(grid, [x0, x1]):
                ax.axis("off")
                ax.imshow(im.squeeze(), cmap="hot")

            pil_img = fig2img(fig)
            save_list.append(pil_img)
            plt.close(fig)
        else:
            img_list = [x0, x1]
            grid_img = make_grid(img_list, nrow=2, normalize=True)
            save_list.append(grid_img)

        count += 1

    wandb_logger.log_image(key="image_pairs", images=save_list)
#-----------------------------------

def rgz_cut(rgz_dset, threshold, mb_cut: bool = True, remove_duplicates=False):
    """Cut rgz data-set based on angular size and whether data-point is contained in MiraBest"""

    n = len(rgz_dset)
    idx_bool = np.ones(n, dtype=bool)
    idx = np.arange(n)

    if remove_duplicates:
        idx_bool = np.zeros(n, dtype=bool)
        _, idx_unique = np.unique(rgz_dset.data, axis=0, return_index=True)
        idx_bool[idx_unique] = True

        print(f"Removed {n - np.count_nonzero(idx_bool)} duplicate samples")
        n = np.count_nonzero(idx_bool)

    idx_bool *= rgz_dset.sizes > threshold
    print(f"Removing {n - np.count_nonzero(idx_bool)} samples below angular size threshold.")
    n = np.count_nonzero(idx_bool)

    if mb_cut:
        idx_bool *= rgz_dset.mbflg == 0

        # Print number of MB samples removed
        print(f"Removed {n - np.count_nonzero(idx_bool)} MiraBest samples from RGZ")

    idx = np.argwhere(idx_bool)

    subset = D.Subset(rgz_dset, idx)
    print(f"RGZ dataset cut from {n} to {len(subset)} samples")
    return subset


def dset2tens(dset):
    """Return a tuple (x, y) containing the entire input dataset (careful with large datasets)"""
    return next(iter(DataLoader(dset, int(len(dset)))))


def compute_mu_sig_images(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        n_channels = next(iter(loader))[0].shape[1]

        # Calculate mean
        mu = torch.zeros(n_channels)
        for x, _ in loader:
            # print(x.shape, _.shape)
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                weight = x.shape[0] / n_dset
                mu[c] += weight * torch.mean(x_c).item()

        # Calculate std
        D_sq = torch.zeros(n_channels)
        for x, _ in loader:
            for c in np.arange(n_channels):
                x_c = x[:, c, :, :]
                D_sq += torch.sum((x_c - mu[c]) ** 2)
        sig = (D_sq / (n_dset * x.shape[-1] * x.shape[-2])) ** 0.5

        mu, sig = tuple(mu.tolist()), tuple(sig.tolist())
        print(f"mu: {mu}, std: {sig}")
        return mu, sig

    else:
        x, _ = dset2tens(dset)
        n_channels = x.shape[1]
        mu, sig = [], []
        for c in np.arange(n_channels):
            x_c = x[:, c, :, :]
            mu.append(torch.mean(x).item())
            sig.append(torch.std(x).item())
        return tuple(mu), tuple(sig)


def compute_mu_sig_features(dset, batch_size=0):
    """Compute mean and standard variance of a dataset (use batching with large datasets)"""
    print("Computing mean and std of dataset")
    if batch_size:
        # Load samples in batches
        n_dset = len(dset)
        loader = DataLoader(dset, batch_size)
        x, _ = next(iter(loader))
        n, c, h, w = x.shape

        # Calculate mean
        mu = 0
        print("Computing mean")
        for x, _ in tqdm(loader):
            x = x
            weight = x.shape[0] / n_dset
            mu += weight * torch.mean(x)

        # Calculate std
        D_sq = 0
        print("Computing std")
        for x, _ in tqdm(loader):
            D_sq += torch.sum((x - mu) ** 2)
        std = (D_sq / (n_dset * h * w)) ** 0.5

        print(f"mean: {mu}, std: {std}")
        return mu, std.item()

    else:
        x, _ = dset2tens(dset)
        return torch.mean(x).item(), torch.std(x).item()


def train_val_test_split(dset, test_size=0.2, val_size=0.2, val_seed=69, test_seed=69):
    """Split a dataset into train, validation and test sets"""

    idx = np.arange(len(dset))
    idx_train_val, idx_test = train_test_split(idx, test_size=test_size, random_state=test_seed)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_size, random_state=val_seed)

    dset_train = D.Subset(dset, idx_train)
    dset_val = D.Subset(dset, idx_val)
    dset_test = D.Subset(dset, idx_test)

    return dset_train, dset_val, dset_test


def iterative_split(idx, *args, seed=69):
    """Return a list of random subsets of idx"""

    generator = np.random.default_rng(seed)
    generator.shuffle(idx)

    split_idx = []
    for i in idx:
        split_idx.append(len(split_idx) + i * (len(idx) - len(split_idx)))

    subset_idx = []
    j = 0
    for i in range(len(split_idx)):
        subset_idx[i] = idx[split_idx[i - 1 : i]]
        j = split_idx[i]
