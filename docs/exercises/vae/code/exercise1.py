from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F

FIGURES = Path(__file__).resolve().parents[1] / "figures" 

learning_rate = 10**-3
batch_size =128
epochs =20

class VAE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256,d)      
        self.fc_logvar = nn.Linear(256,d)  
        self.decoder = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(), nn.Linear(512,784), nn.Sigmoid())  

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)     
    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        z = mu + std * eps
        return z
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def data():
    train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                            generator=torch.Generator().manual_seed(42))
    return test, train, val

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    B = x.size(0)                 # tamanho do batch
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)/B
    return recon + beta * kl, recon, kl

def treino():
    test, train, val = data()     # na ordem em que a sua função devolve!

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)
    model = VAE(d=16)
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    hist = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}

    for epoch in range(epochs):
        model.train()
        soma_recon, soma_kl, n = 0.0, 0.0, 0
        for x, _ in train_loader:
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            soma_recon += recon.item() * x.size(0)
            soma_kl    += kl.item() * x.size(0)
            n += x.size(0)
        hist["train_recon"].append(soma_recon / n)
        hist["train_kl"].append(soma_kl / n)

        val_recon, val_kl = avaliar(model, val_loader)
        hist["val_recon"].append(val_recon)
        hist["val_kl"].append(val_kl)
        print(f"época {epoch+1}: val recon {val_recon:.1f}  val kl {val_kl:.1f}")

    return model, hist, test_loader

def avaliar(model, loader):
    model.eval()
    soma_recon, soma_kl, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x_hat, mu, logvar = model(x)
            _, recon, kl = vae_loss(x_hat, x, mu, logvar)
            soma_recon += recon.item() * x.size(0)
            soma_kl    += kl.item() * x.size(0)
            n += x.size(0)
    return soma_recon / n, soma_kl / n


def figura1(hist):
    ep = range(1, len(hist["train_recon"]) + 1)

    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(11, 4))
    ax_r.plot(ep, hist["train_recon"], marker="o", label="treino")
    ax_r.plot(ep, hist["val_recon"], marker="s", label="validação")
    ax_r.set_title("Reconstrução (BCE)")
    ax_r.set_xlabel("época")
    ax_r.set_ylabel("nats / imagem")
    ax_r.legend()

    ax_k.plot(ep, hist["train_kl"], marker="o", label="treino")
    ax_k.plot(ep, hist["val_kl"], marker="s", label="validação")
    ax_k.set_title("KL")
    ax_k.set_xlabel("época")
    ax_k.set_ylabel("nats / imagem")
    ax_k.legend()

    for ax in (ax_r, ax_k):
        ax.set_xticks(list(ep))
        ax.grid(alpha=0.3)
    fig.suptitle("Figura 1 — Termos da loss por época (d = 16)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-curvas-loss.png", dpi=150)
    plt.close(fig)


def figura2(model, test):
    # primeiro índice do teste para cada dígito 0..9
    idx = [(test.targets == dig).nonzero()[0].item() for dig in range(10)]
    x = torch.stack([test[i][0] for i in idx]).view(10, -1)

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
        x_hat = model.decode(mu)        # decodifica mu, sem sortear z

    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
    for col in range(10):
        axes[0, col].imshow(x[col].view(28, 28), cmap="gray")
        axes[1, col].imshow(x_hat[col].view(28, 28), cmap="gray")
        axes[1, col].set_xlabel(str(col))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstrução")
    fig.supxlabel("dígito")
    fig.suptitle("Figura 2 — Imagens de teste e reconstruções (decodificando μ)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-reconstrucoes.png", dpi=150)
    plt.close(fig)


def figura3(model, d=16):
    torch.manual_seed(0)
    z = torch.randn(64, d)              # z ~ N(0, I)

    model.eval()
    with torch.no_grad():
        x_gen = model.decode(z)

    fig, axes = plt.subplots(8, 8, figsize=(7, 7.4))
    for ax, img in zip(axes.flat, x_gen):
        ax.imshow(img.view(28, 28), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.supxlabel("amostra (coluna)")
    fig.supylabel("amostra (linha)")
    fig.suptitle("Figura 3 — 64 amostras decodificadas de z ~ N(0, I)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-amostras.png", dpi=150)
    plt.close(fig)



def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    modelo, hist, test_loader = treino()
    test_recon, test_kl = avaliar(modelo,test_loader)
    print(f"TESTE: recon {test_recon:.2f}  kl {test_kl:.2f}  "
          f"-ELBO {test_recon + test_kl:.2f}  nats/imagem")
    figura1(hist)
    figura2(modelo, test_loader.dataset)
    figura3(modelo)
    

   
    
    



if __name__ == "__main__":
    main()
