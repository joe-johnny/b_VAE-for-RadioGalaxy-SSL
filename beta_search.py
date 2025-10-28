#importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
# from MiraBest import MiraBest
import torch.nn.functional as F
import pandas as pd
import wandb
import json

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN vae model
class CNNEncoder(nn.Module): #Encoder
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

    def forward(self, x): #encoder forward pass
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        log_var = self.logvar(x)
        return mu, log_var

class CNNDecoder(nn.Module): #Decoder
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 64*9*9)
        self.deconv_layers = nn.Sequential(
            # nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
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
    
    def forward(self, z): #decoder forward pass
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))        
        x = x.view(x.shape[0], 64, 9, 9)
        x = self.deconv_layers(x)
        return x

class CNNVAE(nn.Module): #VAE model
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNVAE, self).__init__()
        
        self.encoder = CNNEncoder(z_dim)
        self.decoder = CNNDecoder(z_dim)
        
    def reparam_trick(self, mu, log_var): #reparameterization trick
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + (epsilon * sigma)
        return z
    
    def encode(self, x): #encoder forward pass
        mu, log_var = self.encoder(x)
        return mu, log_var
    
    def decode(self,z): #decoder forward pass
        return self.decoder(z)

    def forward(self, x): #forward pass
        mu, log_var = self.encoder(x)
        z = self.reparam_trick(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
#vae loss - elbo
def elbo(x_recon, x, mu, log_var, beta):
    
    recon_loss = (F.binary_cross_entropy(x_recon, x, reduction='sum'))/ x.size(0)   #reconstruction loss
    
    kl_loss = (-0.5 * (torch.sum(1 + log_var - mu.pow(2) - log_var.exp())))/ x.size(0)  #kl divergence loss

    total_loss = recon_loss + (beta * kl_loss)  #total loss
    return total_loss, recon_loss, kl_loss

#training model
def train_vae(model, dataloader, beta, epochs):
    # wandb.init(project = "VAE-MScR", name = 'beta_search-rgz')
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #adam optimizers
    model.train()
    # epoch_train_loss = 0
    # epoch_recon_loss = 0
    # epoch_kl_loss = 0

    for epoch in range(epochs):
        total_loss = 0
        model
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            
            loss, recon_loss, kl_loss = elbo(recon_x, x, mu, logvar, beta) #elbo loss
            
            epoch_train_loss += (loss.item() * x.size(0))
            epoch_recon_loss += (recon_loss.item() * x.size(0))
            epoch_kl_loss    += (kl_loss.item() * x.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        dataset_size = len(dataloader.dataset)
        avg_train_loss = epoch_train_loss / dataset_size
        avg_recon_loss = epoch_recon_loss / dataset_size
        avg_kl_loss = epoch_kl_loss / dataset_size

        wandb.log({
            "beta": beta,
            # "learning_rate": lr,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss"   : avg_kl_loss,
        })
        print(f"Epoch {epoch+1}/{epochs}, Beta={beta:.3f}, Loss={avg_train_loss:.4f}, recon_loss={avg_recon_loss:.4f}, kl_loss={avg_kl_loss:.4f}")
    # wandb.finish()
    return model

#disentanglement metric following Higgins(2017)
def disentanglement_metric(model, dataloader, n_pairs = 100):
    model.eval()
    z_diffs = []    # empty list for diff in latent rep
    y_labels = []  # empty list for label match (same -0 or diff -1)

    data = [(x, l) for (x, l) in dataloader.dataset] #loading data, x-img and l-label
    fr1 = [d for d in data if d[1]==1] #list of fr1
    fr2 = [d for d in data if d[1]==2] #list of fr2

    with torch.no_grad():
        for i in range(n_pairs):           #choosing pairs
            if np.random.rand()>0.5:       #same or diff class
                if np.random.rand()>0.5:   #same but fr1 or fr2
                    x1, l1 = fr1[np.random.randint(len(fr1))] #fr1
                    x2, l2 = fr1[np.random.randint(len(fr1))] #fr1
                else:
                    x1, l1 = fr2[np.random.randint(len(fr2))] #fr2
                    x2, l2 = fr2[np.random.randint(len(fr2))] #fr2
                y = 0 #labelled 0 for same class
            else:
                x1,l1 = fr1[np.random.randint(len(fr1))] #fr1
                x2,l2 = fr2[np.random.randint(len(fr2))] #fr2
                y = 1 #labelled 1 for diff classes
            
            x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) #x1 and x2 to device
            mu1, _ = model.encode(x1) #find mu for x1
            mu2, _ = model.encode(x2) #find mu for x2

            z_diff = (mu1 - mu2).cpu().numpy().flatten() #find diff in latent rep
            z_diffs.append(z_diff) #latent rep diff list
            y_labels.append(y) #label list
    
    clf = LogisticRegression(max_iter=1000) #logistic regression
    clf.fit(z_diffs, y_labels) #fitting the model
    accuracy = accuracy_score(y_labels, clf.predict(z_diffs)) #accuracy score
    return accuracy

#latent traversal - gen and visualizes recon imgs by varying 1 zdim at a time
def latent_traversal(model, x, z_dim, beta): 
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x.unsqueeze(0).to(device))
        steps = 5 #steps in latent traversal
        traversal_range = np.linspace(-3, 3, steps) #range of traversal - statistical coverage of -3sigma to +3sigma
        fig, axes = plt.subplots(z_dim, steps, figsize=(steps*2, z_dim*2))

        for i in range(z_dim):
            for j, val in enumerate(traversal_range):
                z = mu.clone()
                z[0,i] = val #
                x_recon = model.decode(z).cpu().numpy().squeeze()
                axes[i,j].imshow(x_recon, cmap = 'hot') #plotting
                axes[i,j].axis('off') #
                if j == 0 : #
                    axes[i,j].set_ylabel(f'z[{i}]', rotation = 0, labelpad = 20)
                if i == 0: #
                    axes[i,j].set_title(f'{val:.1f}') #
            
        beta_str = str(beta).replace('.', '_')
        plt.savefig(f'3lt_beta_{beta_str}.png', bbox_inches='tight')
        plt.close(fig)
         # plt.show()

        wandb.log({"Latent traversal": [wandb.Image(f'3lt_beta_{beta_str}.png')]})

import torch.utils.data as D
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import torchvision.transforms as T
import pytorch_lightning as pl
import torch.utils.data as data

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Tuple, Type, Optional
from torch.utils.data import Subset
from einops import rearrange
from torchvision.transforms.functional import center_crop, resize

# -----------------------------------------------------------------------------------------

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
# DataSet Laoding
# transform = T.ToTensor()         # Convert to tensor

root = os.path.abspath("/share/nas2_3/jalphonse/data/rgz")

# Load training dataset
train_dataset = RGZ108k(
    root=root,
    train=True,
    # transform=transform,
    remove_duplicates=True,
    cut_threshold=0.0,  # Keep all sizes
    mb_cut=False)        # Keep MiraBest samples

# #checking size of train and test data
# print('Train data size :', len(train_loader.dataset))

class MergedRGZDataset(Dataset):
    def __init__(self, original_dataset, df, transform=None):
        """
        Args:
            original_dataset (RGZ108k): The original RGZ108k dataset
            df (pd.DataFrame): DataFrame containing 'rgz_name' column and other data
            label_column (str): Name of the column in df to use as labels
            transform (callable, optional): Optional transform to be applied to images
        """
        self.transform = transform
        # self.label_column = label_column
        
        # Handle duplicates by keeping only the first occurrence
        df_unique = df.drop_duplicates(subset=['rgz_name'], keep='first')
        print(f"Removed {len(df) - len(df_unique)} duplicate rgz_name entries from DataFrame")
        
        # Convert df to dictionary for faster lookup
        self.df_dict = df_unique.set_index('rgz_name').to_dict('index')
        
        # Find matching indices and store data
        self.matching_indices = []
        self.images = []
        self.labels = []
        
        for i, rgzid in enumerate(original_dataset.rgzid):
            if rgzid in self.df_dict:
                self.matching_indices.append(i)
                self.images.append(original_dataset.data[i])
                self.labels.append(self.df_dict[rgzid]['fr prediction'])
        
        print(f"Created merged dataset with {len(self.images)} samples")

    def __getitem__(self, index):
        """
        Returns:
            tuple: (image_tensor, label)
        """
        # Convert image array to PIL Image
        img = self.images[index].squeeze()
        img = Image.fromarray(img, mode="L")
        
        # Convert PIL Image to tensor
        img_tensor = self.transform(img)
        
        # Get the label
        label = self.labels[index]
        
        return img_tensor, label

    def __len__(self):
        return len(self.images)
    
csvfile = './VAE_RadioGalaxies/rgz_data_preds_filtered.csv'
df = pd.read_csv(csvfile)

df2  = df.drop(['Unnamed: 0'], axis=1)
df1 = df2.drop(['fr vote fraction'], axis=1)

transform = transforms.ToTensor()
merged_train_dataset = MergedRGZDataset(train_dataset, df1, transform=transform)

batch_size = 32
train_loader = DataLoader(merged_train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

# Extract all imgs and labels from train_loader
all_imgs = []
all_labels = []

for imgs,labels in train_loader:
    all_imgs.append(imgs)
    all_labels.append(labels)

# Concatenate all batches into single tensors
all_imgs = torch.cat(all_imgs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

#main workflow
def main():
    # beta_values = np.linspace(1.0, 3.0, 20) #range of beta values from 0.8 to 3.8
    beta_values = np.round(np.linspace(1.0, 3.0, 20), 2)
    z_dim = 8 #latent dim
    results = {}
    epochs = 80

    os.makedirs("results", exist_ok=True)
    run_id = os.environ.get("RUN_ID", "default")  # You can also pass this via argparse if not using sbatch
    save_path = f"results/run_{run_id}.json"

    for beta in beta_values:
        beta_str = str(beta).replace('.', '_')
        wandb.init(project="VAE-MScR", name=f'rgz_search_beta_{beta_str}')

        model = CNNVAE(z_dim).to(device) #model initialization
        trained_model = train_vae(model, train_loader, beta, epochs) #training the model

        accuracy = disentanglement_metric(model, train_loader) #disentanglement metric
        results[beta] = accuracy
        wandb.log({'Disentanglement Accuracy' : accuracy})
        print(f'Beta = {beta}, Disentanglement Accuracy = {accuracy:.4f}')

        sample_img = all_imgs[0] #sample image for latent traversal
        latent_traversal(trained_model, sample_img, z_dim, beta) #latent traversal

        wandb.finish()
    
    optimal_beta = max(results, key=results.get) #finding optimal beta value
    print(f'Optimal Beta: {optimal_beta}, Accuracy: {results[optimal_beta]:.4f}')
    
    plt.plot(list(results.keys()), list(results.values())) #plotting beta values vs accuracy
    plt.xlabel('Beta values')
    plt.ylabel('Disentanglement Accuracy')
    plt.title('Disentanglement Accuracy vs Beta values')
    plt.savefig('acc_v_b')

    with open(save_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main() #main function