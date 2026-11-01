from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

FIGURES = Path(__file__).resolve().parents[1] / "figures" 

learning_rate = 10**-3
batch_size =128
epochs =20

# imagens de teste interpoladas na Figura 6 (dígitos diferentes)
IDX_A = 1   # dígito 2
IDX_B = 0   # dígito 7

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

def treino(d):
    test, train, val = data()     # na ordem em que a sua função devolve!

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)
    model = VAE(d=d)
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


def figura4(model, test_loader):
    model.eval()
    mus, ys = [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu, _ = model.encode(x.view(x.size(0), -1))
            mus.append(mu)
            ys.append(y)
    mu = torch.cat(mus).numpy()
    y = torch.cat(ys).numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    cores = plt.get_cmap("tab10")
    for dig in range(10):
        sel = y == dig
        ax.scatter(mu[sel, 0], mu[sel, 1], s=3, alpha=0.6, color=cores(dig), label=str(dig))
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 — μ das 10 000 imagens de teste (d = 2), por dígito")
    ax.legend(title="dígito", markerscale=4, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-espaco-latente.png", dpi=150)
    plt.close(fig)


def figura5(model, n=15):
    q = np.linspace(0.05, 0.95, n)
    z_vals = norm.ppf(q)                 # quantis da N(0, 1): massa igual por célula

    # linha de cima = z2 maior, para o eixo vertical crescer para cima
    z1, z2 = np.meshgrid(z_vals, z_vals[::-1])
    z = torch.tensor(np.stack([z1.ravel(), z2.ravel()], axis=1), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        imgs = model.decode(z).view(n, n, 28, 28).numpy()
    canvas = imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas, cmap="gray")
    centros = np.arange(n) * 28 + 13.5
    ax.set_xticks(centros)
    ax.set_xticklabels([f"{v:.2f}" for v in np.round(z_vals, 2) + 0.0], rotation=90)
    ax.set_yticks(centros)
    ax.set_yticklabels([f"{v:.2f}" for v in np.round(z_vals[::-1], 2) + 0.0])
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    ax.set_title("Figura 5 — Decodificação de uma grade 15×15 de quantis da prior (d = 2)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-grade-latente.png", dpi=150)
    plt.close(fig)


def figura6(model, test, idx_a, idx_b, passos=10):
    x_a = test[idx_a][0].view(1, -1)
    x_b = test[idx_b][0].view(1, -1)
    t = torch.linspace(0, 1, passos).view(-1, 1)

    pixel = (1 - t) * x_a + t * x_b       # mistura direta dos pixels

    model.eval()
    with torch.no_grad():
        mu_a, _ = model.encode(x_a)
        mu_b, _ = model.encode(x_b)
        latente = model.decode((1 - t) * mu_a + t * mu_b)

    fig, axes = plt.subplots(2, passos, figsize=(13, 3.2))
    for col in range(passos):
        axes[0, col].imshow(pixel[col].view(28, 28), cmap="gray")
        axes[1, col].imshow(latente[col].view(28, 28), cmap="gray")
        axes[1, col].set_xlabel(f"{t[col].item():.2f}")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("pixels")
    axes[1, 0].set_ylabel("latente")
    fig.supxlabel("t")
    fig.suptitle(f"Figura 6 — Interpolação entre as imagens de teste {idx_a} "
                 f"(dígito {test.targets[idx_a].item()}) e {idx_b} "
                 f"(dígito {test.targets[idx_b].item()}), modelo d = 16")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-interpolacao.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    modelo1, hist1, test_loader1 = treino(16)
    test_recon1, test_kl1 = avaliar(modelo1,test_loader1)
    print(f"TESTE d=16: recon {test_recon1:.2f}  kl {test_kl1:.2f}  "
          f"-ELBO {test_recon1 + test_kl1:.2f}  nats/imagem")
    modelo2, hist2, test_loader2 = treino(2)
    test_recon2, test_kl2 = avaliar(modelo2,test_loader2)
    print(f"TESTE d=2:  recon {test_recon2:.2f}  kl {test_kl2:.2f}  "
          f"-ELBO {test_recon2 + test_kl2:.2f}  nats/imagem")

    figura4(modelo2, test_loader2)
    figura5(modelo2)
    figura6(modelo1, test_loader1.dataset, IDX_A, IDX_B)


if __name__ == "__main__":
    main()
