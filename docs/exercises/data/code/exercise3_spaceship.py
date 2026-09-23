"""Exercise 3 — Preparing Real-World Data for a Neural Network.

Prepara o Spaceship Titanic para uma rede com ativação ``tanh``: descreve os
dados, separa treino/teste ANTES de qualquer estatística, imputa, codifica,
aplica log(1+x) nas colunas de gasto e padroniza. Salva a Figura 6 em
``figures/`` e imprime as verificações finais.

O ``train.csv`` do Kaggle vive fora de ``docs/`` para não ser publicado junto
com o site: ``data/spaceship-titanic/train.csv`` na raiz do repositório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise3_spaceship.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "data" / "spaceship-titanic" / "train.csv"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42  # mesma semente do resto do relatório

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Name", "Cabin"]
TARGET = "Transported"


def describe(df: pd.DataFrame) -> None:
    """Item A — objetivo, balanceamento, tipos, ausentes e caudas dos gastos."""
    print(f"== Forma do arquivo bruto ==\n  {df.shape}\n")

    balance = df[TARGET].value_counts(normalize=True).sort_index()
    print("== Balanceamento de Transported ==")
    for label, share in balance.items():
        print(f"  {label}: {share:.4%}  ({int(df[TARGET].eq(label).sum())} passageiros)")

    print("\n== Valores ausentes por coluna (markdown) ==")
    missing = pd.DataFrame({
        "Ausentes": df.isna().sum(),
        "%": (df.isna().mean() * 100).round(2),
    })
    missing = missing[missing["Ausentes"] > 0].sort_values("Ausentes", ascending=False)
    print("| Coluna | Ausentes | % do total |")
    print("|---|---:|---:|")
    for col, row in missing.iterrows():
        print(f"| `{col}` | {int(row['Ausentes'])} | {row['%']:.2f}% |")
    print(f"  total de células ausentes: {int(df.isna().sum().sum())}")

    print("\n== Colunas de gasto: média, mediana e máximo (markdown) ==")
    print("| Coluna | Média | Mediana | Máximo | % de zeros |")
    print("|---|---:|---:|---:|---:|")
    for col in SPEND:
        mean, median, top = df[col].mean(), df[col].median(), df[col].max()
        zeros = df[col].eq(0).mean() * 100
        print(f"| `{col}` | {mean:.2f} | {median:.2f} | {top:.0f} | {zeros:.1f}% |")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Item C — engenharia de features que é linha a linha (logo, sem vazamento).

    ``TotalSpend`` usa ``min_count=1``: uma linha com os cinco gastos ausentes
    continua ausente (e cai no imputador) em vez de virar um zero inventado.
    """
    out = df.drop(columns=DROP + [TARGET])
    out["TotalSpend"] = out[SPEND].sum(axis=1, min_count=1)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Imputação + log(1+x) + padronização, nas colunas certas.

    Tudo dentro de um único ``ColumnTransformer`` para que ``fit`` só enxergue o
    treino: mediana, média, desvio-padrão e categorias observadas saem dali.
    """
    spend_pipeline = Pipeline([
        # Mediana em vez de média: as colunas de gasto têm cauda longuíssima e
        # a média seria puxada pelos outliers.
        ("imputer", SimpleImputer(strategy="median")),
        # log(1+x) comprime a cauda; 1+x cobre o zero, que é o valor mais comum.
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    age_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        # Moda: ausente aqui não tem significado próprio, e as colunas têm
        # categoria dominante clara.
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": categoria vista só no teste vira uma linha de
        # zeros em vez de quebrar o transform.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("spend", spend_pipeline, SPEND + ["TotalSpend"]),
        ("age", age_pipeline, ["Age"]),
        ("cat", categorical_pipeline, CATEGORICAL),
    ])


def figure6(raw_train: pd.Series, processed: np.ndarray, column: str) -> None:
    """Figura 6 — uma feature de cauda pesada antes e depois do pré-processamento."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(raw_train.dropna(), bins=50, color="#4c72b0")
    axes[0].set_title(f"{column} — bruto")
    axes[0].set_xlabel("Gasto (créditos)")

    axes[1].hist(np.log1p(raw_train.dropna()), bins=50, color="#dd8452")
    axes[1].set_title(f"{column} — após $\\log(1+x)$")
    axes[1].set_xlabel(r"$\log(1 + $ gasto $)$")

    axes[2].hist(processed, bins=50, color="#55a868")
    axes[2].set_title(f"{column} — após $\\log(1+x)$ + padronização")
    axes[2].set_xlabel("Valor padronizado (média 0, desvio 1)")

    for ax in axes:
        ax.set_ylabel("Passageiros (treino)")
        ax.grid(alpha=0.25)
    fig.suptitle("Figura 6 — Efeito do pré-processamento sobre uma feature de "
                 "cauda pesada", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {CSV}\n"
            "Baixe train.csv em https://www.kaggle.com/competitions/spaceship-titanic"
        )

    df = pd.read_csv(CSV)
    describe(df)

    print(f"\n== Tipos de feature ==\n  numéricas  : {['Age'] + SPEND}")
    print(f"  categóricas: {CATEGORICAL}")
    print(f"  descartadas: {DROP}  (identificadores e texto livre)")

    # --- Item B: o split vem ANTES de qualquer estatística -------------------
    X = engineer(df)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print("\n== Split 80/20 estratificado ==")
    print(f"  treino: {X_train.shape}  | positivos: {y_train.mean():.4%}")
    print(f"  teste : {X_test.shape}  | positivos: {y_test.mean():.4%}")

    print("\n== FoodCourt no TREINO, antes de transformar ==")
    print(f"  média = {X_train['FoodCourt'].mean():.2f} | "
          f"mediana = {X_train['FoodCourt'].median():.2f} | "
          f"máximo = {X_train['FoodCourt'].max():.0f}")

    # --- Item C: fit só no treino, transform nos dois ------------------------
    preprocessor = build_preprocessor()
    Z_train = preprocessor.fit_transform(X_train)   # fit + transform: TREINO
    Z_test = preprocessor.transform(X_test)         # só transform:    TESTE

    names = list(preprocessor.get_feature_names_out())
    print("\n== Matriz de features ==")
    print(f"  treino: {Z_train.shape} | teste: {Z_test.shape}")
    print(f"  {len(names)} colunas a partir das {X_train.shape[1]} originais")
    n_numeric = len(SPEND) + 2  # 5 gastos + TotalSpend + Age
    print(f"  {n_numeric} colunas numéricas + {len(names) - n_numeric} binárias do one-hot")

    # --- Item D: verificações finais -----------------------------------------
    print("\n== Verificações finais ==")
    print(f"  NaN no treino: {int(np.isnan(Z_train).sum())} | "
          f"NaN no teste: {int(np.isnan(Z_test).sum())}")
    print(f"  treino: min = {Z_train.min():.4f} | máx = {Z_train.max():.4f}")
    print(f"  teste : min = {Z_test.min():.4f} | máx = {Z_test.max():.4f}")
    numeric_cols = [i for i, n in enumerate(names) if not n.startswith("cat__")]
    print(f"  colunas numéricas — média = {Z_train[:, numeric_cols].mean():.2e} | "
          f"desvio = {Z_train[:, numeric_cols].std():.4f}")

    index = names.index("spend__FoodCourt")
    figure6(X_train["FoodCourt"], Z_train[:, index], "FoodCourt")
    print(f"\n  Figura 6 salva a partir da coluna '{names[index]}' "
          f"(faixa {Z_train[:, index].min():.2f} a {Z_train[:, index].max():.2f})")


if __name__ == "__main__":
    main()
