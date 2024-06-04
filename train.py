import pytorch_lightning as pl
import torch
import numpy as np

from pl_bolts.datamodules import CIFAR10DataModule
from matplotlib.pyplot import imshow, figure
from torchvision.utils import make_grid
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from VAE.VariationalAutoencoder import VAE

cifar_10 = CIFAR10DataModule(batch_size=128)

pl.seed_everything(1234)

vae = VAE()
trainer = pl.Trainer(gpus=1, max_epochs=30, accelerator='cpu')
trainer.fit(vae, cifar_10)

figure(figsize=(8, 3), dpi=300)

num_preds = 16
p = torch.distributions.Normal(torch.zeros([1,256]), torch.ones([1,256]))  #mu and std
z = p.rsample((num_preds,))

with torch.no_grad():
    pred = vae.decoder(z.to(vae.device)).cpu()

# UNDO DATA NORMALIZATION
normalize = cifar10_normalization()
mean, std = np.array(normalize.mean), np.array(normalize.std)
img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

imshow(img);