#VAE RGZ

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision
from torchvision.utils import save_image
from PIL import Image
from torchsummary import summary
from MiraBest import MiraBest_full, MBFRConfident, MBFRUncertain, MBHybrid
import wandb
import random
from tqdm import tqdm

wandb.login(key='your_key')

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.utils.data as D
import os
import sys
import torchvision.transforms as T
import pytorch_lightning as pl

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.utils import download_url, check_integrity

from torch.utils.data import random_split

# -----------------------------------------------------------------------------------------

#RGZ dataset
class RGZ108k(D.Dataset):
    """`RGZ 108k <>`_Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "rgz108k-batches-py"

    # Need to upload this, for now download is commented out
    # url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz108k-batches-python.tar.gz"
    tgz_md5 = "3fef587aa2aa3ece3b01b125977ae19d"
    train_list = [
        ["data_batch_1", "3f0c0eefdfafc0c5b373ad82c6cf9e38"],
        ["data_batch_2", "c39657f335fa8957e9e7cfe35b3503fe"],
        ["data_batch_3", "711cb401d9a039ad90ee61f652361a7e"],
        ["data_batch_4", "2d46b5031e9b8220886124876cb8b426"],
        ["data_batch_5", "8cb6644a59368abddc91af006fd67160"],
        ["data_batch_6", "b523102c655c44e575eb1ccae8af3a56"],
        ["data_batch_7", "58da5e781a566c331e105799d35b801c"],
        ["data_batch_8", "cdab6dd1245f9e91c2e5efb00212cd04"],
        ["data_batch_9", "20ef83c0c07c033c2a1b6b0e028a342d"],
        ["data_batch_10", "dd4e59f515b1309bbf80488f7352c2a6"],
        ["data_batch_11", "23d4e845d685c8183b4283277cc5ed72"],
        ["data_batch_12", "7905d89ba2ef1bc722e4d45357cc5562"],
        ["data_batch_13", "753ce85f565a72fa0c2aaa458a6ea5e0"],
        ["data_batch_14", "4145e21c48163d593eac403fdc259c5d"],
        ["data_batch_15", "713b1f15328e58c210a815affc6d4104"],
        ["data_batch_16", "bd45f4895bed648f20b2b2fa5d483281"],
        ["data_batch_17", "e8fe6c5f408280bd122b64eb1bbc9ad0"],
        ["data_batch_18", "1b35a3c4da301c7899356f890f8c08af"],
        ["data_batch_19", "357af43d0c18b448d38f37d1390d194e"],
        ["data_batch_20", "c908a88e9f62975fabf9e2241fe0a02b"],
        ["data_batch_21", "231b1413a2f0c8fda02c496c0b0d9ffb"],
        ["data_batch_22", "8f1b27f220f5253d18da1a4d7c46cc91"],
        ["data_batch_23", "6008ce450b4a4de0f81407da811e6fbf"],
        ["data_batch_24", "180c351fd32c3b204cac17e2fac7b98d"],
        ["data_batch_25", "51be04715b303da51cbe3640a164662b"],
        ["data_batch_26", "9cb972ae3069541dc4fa096ea95149eb"],
        ["data_batch_27", "065d888e4b131485f0a54089245849df"],
        ["data_batch_28", "d0430812428aefaabcec8c4cd8f0a838"],
        ["data_batch_29", "221bdd97fa36d0697deb13e4f708b74f"],
        ["data_batch_30", "81eaec70f17f7ff5f0c7f3fbc9d4060c"],
        ["data_batch_31", "f6ccddbf6122c0bac8befb7e7d5d386e"],
        ["data_batch_32", "e7cdf96948440478929bc0565d572610"],
        ["data_batch_33", "940d07f47d5d98f4a034d2dbc7937f59"],
        ["data_batch_34", "a5c97a274671c0536751e1041a05c0a9"],
        ["data_batch_35", "d4dbb71e9e92b61bfde9a2d31dfb6ec8"],
        ["data_batch_36", "208ef8426ce9079d65a215a9b89941bc"],
        ["data_batch_37", "60d0ca138812e1a8e2d439f5621fa7f6"],
        ["data_batch_38", "b17ff76a0457dc47e331668c34c0e7e6"],
        ["data_batch_39", "28712e629d7a7ceba527ba77184ee9c5"],
        ["data_batch_40", "a9b575bb7f108e63e4392f5dd1672d31"],
        ["data_batch_41", "3390460da44022c13d24f883556a18eb"],
        ["data_batch_42", "7297ca4b77c6059150f471969ca3827a"],
        ["data_batch_43", "0d0e610231994ff3663c662f4b960340"],
        ["data_batch_44", "386a2d3472fbd97330bb7b8bb7e0ff2f"],
        ["data_batch_45", "1124b3bbbe0c7f9c14f964c4533bd565"],
        ["data_batch_46", "18a53af11a51c44632f4ce3c0b012e5c"],
        ["data_batch_47", "05e6a4d27381dcd505e9bea7286929a6"],
        ["data_batch_48", "2c666e471cbd0b547d72bfe0aba04988"],
        ["data_batch_49", "1fde041df048985818326d4f587126c9"],
        ["data_batch_50", "8f2f127fab28d83b8b9182119db16732"],
        ["data_batch_51", "30b39c698faca92bc1a7c2a68efad3e8"],
        ["data_batch_52", "e9866820972ed2b23a46bea4cea1afd8"],
        ["data_batch_53", "379b92e4ad1c6128ec09703120a5e77f"],
    ]

    test_list = [["test_batch", "6c42ba92dc3239fd6ab5597b120741a0"]]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "d5d3d04e1d462b02b69285af3391ba25",
    }

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        remove_duplicates: bool = True,
        cut_threshold: float = 0.0,
        mb_cut=False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.remove_duplicates = remove_duplicates
        self.cut_threshold = cut_threshold
        self.mb_cut = mb_cut

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []  # object image data
        self.names = []  # object file names
        self.rgzid = []  # object RGZ ID
        self.mbflg = []  # object MiraBest flag
        self.sizes = []  # object largest angular sizes

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                # print(entry.keys())

                self.data.append(entry["data"])
                self.names.append(entry["filenames"])
                self.rgzid.append(entry["src_ids"])
                self.mbflg.append(entry["mb_flag"])
                self.sizes.append(entry["LAS"])

        self.rgzid = np.vstack(self.rgzid).reshape(-1)
        self.sizes = np.vstack(self.sizes).reshape(-1)
        self.mbflg = np.vstack(self.mbflg).reshape(-1)
        self.names = np.vstack(self.names).reshape(-1)

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

        # Make cuts on the data
        n = self.__len__()
        idx_bool = np.ones(n, dtype=bool)

        if self.remove_duplicates:
            print(f"Removing duplicates from RGZ dataset...")
            idx_bool = np.zeros(n, dtype=bool)
            _, idx_unique = np.unique(self.data, axis=0, return_index=True)
            idx_bool[idx_unique] = True

            print(f"Removed {n - np.count_nonzero(idx_bool)} duplicate samples")
            n = np.count_nonzero(idx_bool)

        idx_bool *= self.sizes > self.cut_threshold
        print(f"Removing {n - np.count_nonzero(idx_bool)} samples below angular size threshold.")
        n = np.count_nonzero(idx_bool)

        if mb_cut:
            idx_bool *= self.mbflg == 0

            # Print number of MB samples removed
            print(f"Removed {n - np.count_nonzero(idx_bool)} MiraBest samples from RGZ")

        idx = np.argwhere(idx_bool).squeeze()

        self.data = self.data[idx]
        self.names = self.names[idx]
        self.rgzid = self.rgzid[idx]
        self.mbflg = self.mbflg[idx]
        self.sizes = self.sizes[idx]

        print(self.data.shape)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")

            self.classes = data[self.meta["key"]]

        # self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index].squeeze()
        mbf = self.mbflg[index].squeeze()
        rgz = self.rgzid[index].squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = np.reshape(img, (150, 150))
        img = img.squeeze()
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # return img, self.sizes[index].squeeze()

        return img, {"size": las, "mb": mbf, "id": rgz, "index": index}

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

def get_from_id(self, id):
        index = np.argwhere(self.rgzid.squeeze() == id).squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.data[index]
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img

#Plotting and image gen functions

#loss curve per epoch
def plot_elbocurve(train_elbo, test_elbo, latent_size):
    plt.plot(train_elbo, color='b', linestyle='-', label='train')
    plt.plot(test_elbo, color='r', linestyle='-', label='test')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Lower Bound')
    plt.title('RGZ, $N_z$={}'.format(latent_size))
    plt.savefig('vae_elbocurve-{}D.png'.format(latent_size))

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Recon_images
def generate_variance_comparison(model, test_loader, m, save_path):
    wandb.init(project="VAE-MScR", name="cnn-cvae-mb-var-compare", reinit=True)

    model.eval()

    # --- Collect all test images into memory (safe if dataset not too big) ---
    all_data = []
    for batch in test_loader:
        x, _ = batch
        all_data.extend(x.cpu().numpy())
    all_data = np.array(all_data)

    # --- Pick one random source image ---
    random_idx = random.choice(range(len(all_data)))
    source_img = torch.tensor(all_data[random_idx]).unsqueeze(0)  # shape [1, C, H, W]
    source_img = source_img.to(next(model.parameters()).device)   # send to same device as model

    # --- Define variances for N(0, variance) ---
    variances = [0.1, 0.5, 1.0]
    n = len(variances)  # number of rows
    generated_images = {var: [] for var in variances}

    with torch.no_grad():
        mu, logvar = model.encode(source_img)
        for var in variances:
            std = torch.sqrt(torch.tensor(var, dtype=mu.dtype, device=mu.device))
            for j in range(m):
                eps = torch.randn_like(mu) * std
                z = mu + eps
                x_recon = model.decode(z)
                generated_images[var].append(x_recon.squeeze().cpu().numpy())

    # --- Plot ---
    fig, axes = plt.subplots(n, m + 1, figsize=(15, 5 * n))

    if n == 1:  # handle single-row case
        axes = np.expand_dims(axes, 0)

    for i, var in enumerate(variances):
        # Original (same for all rows)
        axes[i, 0].imshow(all_data[random_idx].squeeze(), cmap='magma')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Generations
        for j in range(m):
            gen_img = generated_images[var][j]
            axes[i, j + 1].imshow(gen_img.reshape(150, 150), cmap='magma')
            axes[i, j + 1].set_title(f'N(0,{var}), Gen {j+1}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    wandb.log({"variance_comparison": [wandb.Image(save_path)]})
    wandb.finish()

#images_sampled_from_prior
def generate_prior_samples(model, z_dim, m, save_path):
    wandb.init(project="VAE-MScR", name="cnn-cvae-prior-compare", reinit=True)

    model.eval()

    # --- Define variances for N(0, variance) ---
    variances = [0.1, 0.5, 1.0]
    n = len(variances)  # number of rows
    generated_images = {var: [] for var in variances}

    device = next(model.parameters()).device  # infer device from model

    with torch.no_grad():
        for var in variances:
            std = torch.sqrt(torch.tensor(var, dtype=torch.float32, device=device))
            for j in range(m):
                z = torch.randn((1, z_dim), device=device) * std  # z ~ N(0, var)
                x_recon = model.decode(z)
                generated_images[var].append(x_recon.squeeze().cpu().numpy())

    # --- Plot ---
    fig, axes = plt.subplots(n, m, figsize=(15, 5 * n))

    if n == 1:  # handle single-row case
        axes = np.expand_dims(axes, 0)

    for i, var in enumerate(variances):
        for j in range(m):
            gen_img = generated_images[var][j]
            axes[i, j].imshow(gen_img.reshape(150, 150), cmap='magma')
            axes[i, j].set_title(f'N(0,{var}), gen {j+1}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    wandb.log({"prior_samples": [wandb.Image(save_path)]})
    wandb.finish()


# CNNVae model
class CNNEncoder(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.fc = nn.Linear(64*9*9, 128)
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        log_var = self.logvar(x)
        return mu, log_var

class CNNDecoder(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 64*9*9)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))        
        x = x.view(x.shape[0], 64, 9, 9)
        x = self.deconv_layers(x)
        return x

class CNNVAE(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNVAE, self).__init__()
        self.encoder = CNNEncoder(z_dim)
        self.decoder = CNNDecoder(z_dim)
        
    def reparam_trick(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + (epsilon * sigma)
        return z
    
    def encode(self, x): #encoder forward pass
        mu, log_var = self.encoder(x)
        return mu, log_var
    
    def decode(self,z): #decoder forward pass
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparam_trick(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
# DataSet Loading
transform = transforms.ToTensor()       
root = os.path.abspath("path")

# Load training dataset
train_dataset = RGZ108k(
    root=root,
    train=True,
    transform=transform,
    remove_duplicates=True,
    cut_threshold=20.0,  
    mb_cut=False)

# Load test dataset
test_dataset = RGZ108k(root=root, train=False, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader  = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)

#Part of RGZ dataset with labels from Nuthawara Buthaisong (JBCA) used for test beta search and UMAP. not used for training.

# class MergedRGZDataset(Dataset):
#     def __init__(self, original_dataset, df, transform=None):
        
#         self.transform = transform
        
#         # Handle duplicates by keeping only the first occurrence
#         df_unique = df.drop_duplicates(subset=['rgz_name'], keep='first')
#         print(f"Removed {len(df) - len(df_unique)} duplicate rgz_name entries from DataFrame")
        
#         # Convert df to dictionary for faster lookup
#         self.df_dict = df_unique.set_index('rgz_name').to_dict('index')
        
#         # Find matching indices and store data
#         self.matching_indices = []
#         self.images = []
#         self.labels = []
        
#         for i, rgzid in enumerate(original_dataset.rgzid):
#             if rgzid in self.df_dict:
#                 self.matching_indices.append(i)
#                 self.images.append(original_dataset.data[i])
#                 self.labels.append(self.df_dict[rgzid]['fr prediction'])
        
#         print(f"Created merged dataset with {len(self.images)} samples")

#     def __getitem__(self, index):
#         """
#         Returns:
#             tuple: (image_tensor, label)
#         """
#         # Convert image array to PIL Image
#         img = self.images[index].squeeze()
#         img = Image.fromarray(img, mode="L")
        
#         # Convert PIL Image to tensor
#         img_tensor = self.transform(img)
        
#         # Get the label
#         label = int(self.labels[index]) - 1 
        
#         return img_tensor, label

#     def __len__(self):
#         return len(self.images)
    
# csvfile = './VAE_RadioGalaxies/rgz_data_preds_filtered.csv'
# df = pd.read_csv(csvfile)

# df2  = df.drop(['Unnamed: 0'], axis=1)
# df1 = df2.drop(['fr vote fraction'], axis=1)

# transform = transforms.ToTensor()
# merged_train_dataset = MergedRGZDataset(train_dataset, df1, transform=transform)

val_fraction = 0.2  # 20% validation
n_total = len(train_dataset)
n_val = int(val_fraction * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Parameters
lr = 1e-4
in_dim = 150*150
z_dim = 8

#model
model = CNNVAE(z_dim = z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, capturable=True)

#ELBO function
def ELBO(x, x_recon, mean, log_var, beta):

    # recon loss: total per image
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)

    # kl loss: total per image
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0) #kl div pushes towards gaussian prior

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# model training
wandb.init(project = "VAE-MScR", name = 'cnn-vae-rgz')

epochs = 150
train_loss = []
test_loss  = []

for epoch in range(epochs):
    max_beta = 2.3
    beta = min(1.0, (epoch + 1) / 10.0) * max_beta #Kl_annealing for stable recons

    model.train()
    epoch_train_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    
    for i, (x, _) in enumerate(train_loader):
        
        x = x.to(device)
        
        x_recon, mean, log_var = model(x)
        
        loss, recon_loss, kl_loss = ELBO(x, x_recon, mean, log_var, beta)
        
        batch_size = x.size(0)
        epoch_train_loss += loss.item() * batch_size
        epoch_recon_loss += recon_loss.item() * batch_size
        epoch_kl_loss    += kl_loss.item() * batch_size
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    dataset_size = len(train_loader.dataset)
    avg_train_loss = epoch_train_loss / dataset_size
    avg_recon_loss = epoch_recon_loss / dataset_size
    avg_kl_loss = epoch_kl_loss / dataset_size
    train_loss.append(avg_train_loss)
 
    # Eval
    model.eval()
    with torch.no_grad():
        total_test_elbo = 0
        for i, (x, _ ) in enumerate(test_loader):
            x = x.to(device)
            x_recon, mean, log_var = model(x)
            
            test_elbo, _, _ = ELBO(x, x_recon, mean, log_var, beta)
            total_test_elbo += test_elbo.item() * x.size(0)
            
        avg_test_elbo = total_test_elbo / len(test_loader.dataset)
        test_loss.append(avg_test_elbo)

    print(f"Epoch {epoch+1}/{epochs}, Beta={beta:.3f}, Train Loss={avg_train_loss:.4f}, "
      f"Recon Loss={avg_recon_loss:.4f}, KL Loss={avg_kl_loss:.4f}, Test Loss={avg_test_elbo:.4f}, "
      f"Log Var Mean={log_var.mean().item():.4f}, Log Var Min={log_var.min().item():.4f}")
    
    #logging to wandb
    wandb.log({
        "beta": beta,
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "test_loss": avg_test_elbo,
        "recon_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
        "log_var_mean": log_var.mean().item(),
        "log_var_min": log_var.min().item(),
    })

torch.save(model.state_dict(), f"cnn_rgz_zdim{z_dim}.pt")
print("Training complete.")
wandb.finish()

#Results graph
train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
plot_elbocurve(train_loss, test_loss, z_dim)

#loading for ckpt
# z_dim = 8
# model = CNNVAE(z_dim=z_dim)
# vae_ckpt_path = "path to ckpt.pt"
# model.load_state_dict(torch.load(vae_ckpt_path))
# model.to(device)

#gen imgs
generate_variance_comparison(model, val_loader, m=5, save_path="vae_recon.png")
generate_prior_samples(model, z_dim=8, m=5, save_path="generated_samples.png")

# --------------------
#Code for latent traversal

# Extract all imgs and labels from train_loader
all_imgs = []


for imgs, _ in train_loader:
    all_imgs.append(imgs)
   

# Concatenate all batches into single tensors
all_imgs = torch.cat(all_imgs, dim=0)

sample_img = all_imgs[13]

#latent traversal function
def latent_traversal(model, x, z_dim): 
    with torch.no_grad():
        mu, _ = model.encode(x.unsqueeze(0).to(device))
        steps = 7 #steps in latent traversal
        traversal_range = np.linspace(-3, 3, steps) #range of traversal
        fig, axes = plt.subplots(z_dim, steps, figsize=(steps*2, z_dim*2))

        for i in range(z_dim):
            for j, val in enumerate(traversal_range):
                z = mu.clone()
                z[0,i] = val
                # z[0,6] = 0.0
                x_recon = model.decode(z).cpu().numpy().squeeze()
                axes[i,j].imshow(x_recon, cmap = 'magma') #plotting
                axes[i,j].axis('off') #
                if j == 0 : #
                    axes[i,j].set_ylabel(f'z[{i}]', rotation = 0, labelpad = 20)
                if i == 0: #
                    axes[i,j].set_title(f'{val:.1f}') #
            
        
        plt.savefig('lt_rgz8.png', bbox_inches='tight')
        plt.close(fig)

latent_traversal(model, sample_img, z_dim)

#complete