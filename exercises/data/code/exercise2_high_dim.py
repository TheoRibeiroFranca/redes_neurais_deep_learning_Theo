"""Exercise 2 — Non-Linearity in Higher Dimensions.

Constrói os dois datasets 5D do enunciado — gaussianas deslocadas (Dataset I) e
cascas concêntricas (Dataset II) —, salva as Figuras 4 e 5 em ``figures/`` e
imprime as variâncias explicadas, as distâncias entre centros e a acurácia do
separador radial proposto no item D.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise2_high_dim.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # mesma semente do Exercício 1

N_PER_CLASS = 500
DIM = 5

# --- Dataset I: gaussianas multivariadas deslocadas -------------------------
MU_A = np.zeros(DIM)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# --- Dataset II: cascas concêntricas ----------------------------------------
# O enunciado escreve rho ~ N(2.0, 0.4) e N(5.0, 0.4); 0.4 é lido como
# desvio-padrão (e não variância) — a leitura está declarada no relatório.
RADIUS = {"C (núcleo)": (2.0, 0.4), "D (casca)": (5.0, 0.4)}

COLORS = ("#1f77b4", "#d62728")


def dataset_i() -> tuple[np.ndarray, np.ndarray]:
    """500 pontos por classe de duas normais multivariadas com covariâncias distintas."""
    a = RNG.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    b = RNG.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    return np.vstack([a, b]), np.repeat([0, 1], N_PER_CLASS)


def _unit_directions(n: int) -> np.ndarray:
    """Direções uniformes na esfera unitária de R^5.

    Amostrar v ~ N(0, I_5) e normalizar é o truque padrão: a normal isotrópica
    não privilegia direção alguma, então v/||v|| cai uniformemente na esfera.
    Sortear cada coordenada em [-1, 1] e normalizar concentraria os pontos nas
    diagonais do cubo.
    """
    v = RNG.normal(size=(n, DIM))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dataset_ii() -> tuple[np.ndarray, np.ndarray]:
    """Duas cascas esféricas concêntricas: x = rho * u, com rho gaussiano."""
    blocks = []
    for mean, std in RADIUS.values():
        u = _unit_directions(N_PER_CLASS)
        rho = RNG.normal(mean, std, size=(N_PER_CLASS, 1))
        blocks.append(rho * u)
    return np.vstack(blocks), np.repeat([0, 1], N_PER_CLASS)


def center_distance(X: np.ndarray, y: np.ndarray) -> float:
    """||mu_1 - mu_2|| calculada em 5D, a partir das médias amostrais."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def project(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projeta 5D -> 2D via PCA e devolve também a variância explicada por componente."""
    pca = PCA(n_components=2)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def radial_separator(X: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Acurácia de g(x) = ||x||^2 - t^2: uma única feature quadrática separa o Dataset II."""
    predicted = (np.sum(X ** 2, axis=1) > threshold ** 2).astype(int)
    return float((predicted == y).mean())


def figure4(projections: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
    """Figura 4 — projeções PCA dos dois datasets, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, (Z, ratio, labels)) in zip(axes, projections.items()):
        for k, (label, color) in enumerate(zip(labels, COLORS)):
            mask = np.repeat([0, 1], N_PER_CLASS) == k
            ax.scatter(*Z[mask].T, s=12, alpha=0.6, color=color, label=f"Classe {label}")
        ax.set_xlabel(f"PC1 ({ratio[0]:.1%} da variância)")
        ax.set_ylabel(f"PC2 ({ratio[1]:.1%} da variância)")
        ax.set_title(f"{name}\nPC1 + PC2 = {ratio.sum():.1%} da variância")
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Figura 4 — Projeção PCA (5D → 2D) dos dois datasets", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-projections.png", dpi=150)
    plt.close(fig)


def figure5(datasets: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
    """Figura 5 — histograma do raio ||x||, as duas classes sobrepostas, por dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, (X, y, labels)) in zip(axes, datasets.items()):
        radii = np.linalg.norm(X, axis=1)
        bins = np.linspace(radii.min(), radii.max(), 45)
        for k, (label, color) in enumerate(zip(labels, COLORS)):
            ax.hist(radii[y == k], bins=bins, alpha=0.6, color=color,
                    label=f"Classe {label}")
        ax.set_xlabel(r"Raio $\|x\|$")
        ax.set_ylabel("Frequência")
        ax.set_title(name)
        ax.legend(loc="upper right")
    fig.suptitle("Figura 5 — Distribuição do raio em 5D, por classe", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-histograms.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X1, y1 = dataset_i()
    X2, y2 = dataset_ii()
    labels1, labels2 = ["A", "B"], ["C (núcleo)", "D (casca)"]

    Z1, ratio1 = project(X1)
    Z2, ratio2 = project(X2)

    figure4({
        "Dataset I — gaussianas deslocadas": (Z1, ratio1, labels1),
        "Dataset II — cascas concêntricas": (Z2, ratio2, labels2),
    })
    figure5({
        "Dataset I — gaussianas deslocadas": (X1, y1, labels1),
        "Dataset II — cascas concêntricas": (X2, y2, labels2),
    })

    print("== Formas ==")
    print(f"  Dataset I : {X1.shape}   Dataset II: {X2.shape}")

    print("\n== Distância entre os centros (em 5D) ==")
    print(f"  Dataset I : {center_distance(X1, y1):.4f}")
    print(f"  Dataset II: {center_distance(X2, y2):.4f}")

    print("\n== Variância explicada pela PCA ==")
    for name, ratio in (("Dataset I ", ratio1), ("Dataset II", ratio2)):
        print(f"  {name}: PC1 = {ratio[0]:.2%} | PC2 = {ratio[1]:.2%} | "
              f"PC1+PC2 = {ratio.sum():.2%}")

    print("\n== Raio médio por classe (5D) ==")
    for name, X, y, labels in (("Dataset I ", X1, y1, labels1),
                               ("Dataset II", X2, y2, labels2)):
        radii = np.linalg.norm(X, axis=1)
        print(f"  {name}: {labels[0]} = {radii[y == 0].mean():.3f} | "
              f"{labels[1]} = {radii[y == 1].mean():.3f}")

    # Item D: o separador não-linear explícito. O ponto médio entre os raios
    # médios (3.5) é o limiar natural, e uma única feature quadrática basta.
    threshold = 3.5
    print(f"\n== Separador radial g(x) = ||x||² - {threshold}² no Dataset II ==")
    print(f"  acurácia = {radial_separator(X2, y2, threshold):.2%}")


if __name__ == "__main__":
    main()
