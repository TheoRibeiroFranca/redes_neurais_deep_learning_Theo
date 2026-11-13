# Figuras 2 a 7 da entrega de regressão (BTC/USD 1 min). Cada função lê results/ ou cache/ e grava em figures/.
# Aula 7 (Regularization), "Reading the two curves": curvas de treino e validação por época (fig2, fig3).
# Aula 8.2 (Regression), "The number that four different models share": previsto x real e resíduo x previsto (fig4, fig5).
# Aula 8.2, "Heteroscedastic": erro por segmento (fig6). RMSE/MAE como alarme (fig5).
# Importância por permutação (fig7): sklearn permutation_importance e Breiman (2001). Não é de aula, exigido pela rubrica.
# Gráficos: matplotlib, figsize (7, 4.5), dpi 150, título, eixos com unidade, legenda.
import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CODE = Path(__file__).resolve().parent
REGRESSION = CODE.parent
RES = REGRESSION / "results"
FIG = REGRESSION / "figures"
CACHE = CODE.parents[3] / "data" / "cache"  # R\data\cache (R = raiz do repo)

FIGSIZE = (7, 4.5)
DPI = 150
Q_LO, Q_HI = 0.001, 0.999   # recorte dos eixos: quantis 0,1% e 99,9%
SPLIT_TXT = {"test": "no teste", "val": "na validação"}


class Faltando(Exception):
    """Entrada ainda não gerada. A figura é pulada e o aviso vai para o console."""


def _num(v):
    """Número em PT-BR (vírgula decimal), para rótulos."""
    return f"{v:g}".replace(".", ",")


def _inteiro(n):
    """Inteiro com separador de milhar de PT-BR."""
    return f"{int(n):,}".replace(",", ".")


def _pct(x):
    """Percentual com vírgula decimal."""
    return f"{x:.2%}".replace(".", ",")


def sci_fmt(v, _pos=None):
    """Eixo de log-retorno em notação científica, vírgula decimal (ex.: -1,5e-03)."""
    return f"{v:.1e}".replace(".", ",")


def _sci_axis(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))
    if y:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_fmt))


def _novo_fig():
    return plt.subplots(figsize=FIGSIZE, dpi=DPI)


def _salvar(fig, out_dir, nome):
    out_dir.mkdir(parents=True, exist_ok=True)
    caminho = out_dir / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)
    return caminho


def _ler_json(caminho, chaves):
    if not caminho.exists():
        raise Faltando(f"falta {caminho}")
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    faltam = [k for k in chaves if k not in dados]
    if faltam:
        raise ValueError(f"{caminho.name} sem as chaves {faltam}")
    return dados


def _ler_preds(caminho):
    """Lê o npz de previsões (formato do contrato). Descarta linhas com NaN."""
    if not caminho.exists():
        raise Faltando(f"falta {caminho}")
    z = np.load(caminho)
    for k in ("y", "yhat"):
        if k not in z.files:
            raise ValueError(f"{caminho.name} sem a chave {k}")
    y = z["y"].astype(np.float64)
    yhat = z["yhat"].astype(np.float64)
    ok = np.isfinite(y) & np.isfinite(yhat)
    split = str(z["split"]) if "split" in z.files else "test"
    return y[ok], yhat[ok], split


def _janela(v):
    """Limites nos quantis 0,1% e 99,9% do vetor."""
    lo, hi = np.quantile(v, [Q_LO, Q_HI])
    return float(lo), float(hi)


def _rotulo_perda(cfg):
    if cfg.get("loss") == "huber":
        base = "Huber, δ = " + _num(cfg.get("huber_delta", 1.0))
    else:
        base = "MSE"
    return f"{base} sobre y/σ, adimensional"


def _rotulo_reg(cfg):
    partes = []
    if cfg.get("weight_decay", 0) > 0:
        partes.append(f"decaimento de pesos = {_num(cfg['weight_decay'])}")
    if cfg.get("dropout", 0) > 0:
        partes.append(f"dropout = {_num(cfg['dropout'])}")
    nome = " + ".join(partes) if partes else "sem regularização"
    return f"{nome} ({cfg.get('name', '?')})"


# ---------------------------------------------------------------- fig2
def fig2_curvas(c):
    """Perda de treino e de validação por época, do modelo final (seed 42), com a melhor época marcada."""
    run = _ler_json(c.res / "runs" / "final_s42.json", ["history", "best_epoch", "cfg"])
    h = run["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = _novo_fig()
    ax.plot(ep, [r["train_loss"] for r in h], marker="o", ms=3, label="treino (conjunto inteiro, modo eval)")
    ax.plot(ep, [r["val_loss"] for r in h], marker="o", ms=3, label="validação")
    ax.axvline(run["best_epoch"], color="gray", ls="--",
               label=f"melhor época ({run['best_epoch']}), pesos restaurados")
    ax.set_xlabel("Época")
    ax.set_ylabel(f"Perda ({_rotulo_perda(run['cfg'])})")
    ax.set_title("Curvas de perda do modelo final (seed 42)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
    return _salvar(fig, c.out, "fig2_curvas.png")


# ---------------------------------------------------------------- fig3
RUNS_4A = ["reg_none", "reg_wd", "reg_dropout", "reg_wd_dropout"]


def fig3_regularizacao(c):
    """Perda de validação por época das quatro configurações da grade 4A (regularização)."""
    fig, ax = _novo_fig()
    feitas, faltam = [], []
    for nome in RUNS_4A:
        cam = c.res / "runs" / f"{nome}.json"
        try:
            run = _ler_json(cam, ["history", "best_epoch", "cfg"])
        except Faltando:
            faltam.append(nome)
            continue
        h = run["history"]
        ep = [r["epoch"] for r in h]
        va = [r["val_loss"] for r in h]
        linha, = ax.plot(ep, va, marker="o", ms=3, label=_rotulo_reg(run["cfg"]))
        b = run["best_epoch"]
        ax.plot([b], [va[b - 1]], marker="*", ms=12, color=linha.get_color(), ls="none")
        feitas.append(nome)
    if not feitas:
        raise Faltando("nenhum run de 4A em runs/ (rode search.py)")
    if faltam:
        print(f"  fig3: aviso, faltam runs {faltam}; plotando {feitas}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de validação (Huber sobre y/σ, adimensional)")
    ax.set_title("Regularização na rede base: validação por época (grade 4A)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(title="estrela = melhor época", fontsize=8)
    return _salvar(fig, c.out, "fig3_regularizacao.png")


# ---------------------------------------------------------------- fig4 / fig5
def fig4_previsto_real(c):
    """Hexbin de previsto x real, com y = x e eixos recortados em 0,1%-99,9%."""
    y, yhat, split = _ler_preds(c.preds)
    lo, hi = _janela(np.concatenate([y, yhat]))
    dentro = (y >= lo) & (y <= hi) & (yhat >= lo) & (yhat <= hi)
    fora = int((~dentro).sum())
    fig, ax = _novo_fig()
    hb = ax.hexbin(yhat[dentro], y[dentro], gridsize=70, extent=(lo, hi, lo, hi),
                   bins="log", mincnt=1, cmap="viridis", linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="red", ls="--", lw=1.2, label="y = ŷ (previsão perfeita)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _sci_axis(ax)
    ax.set_xlabel("Previsto ŷ (log-retorno, adimensional)")
    ax.set_ylabel("Real y (log-retorno, adimensional)")
    ax.set_title(f"Previsto × real {SPLIT_TXT.get(split, split)} (n = {_inteiro(len(y))})")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("pontos por célula (escala log)")
    ax.legend(loc="upper left", fontsize=8,
              title=f"recorte 0,1%–99,9%: {_inteiro(fora)} pontos fora ({_pct(fora / len(y))})")
    return _salvar(fig, c.out, "fig4_previsto_real.png")


def fig5_residuos(c):
    """Hexbin do resíduo e = y - ŷ contra ŷ, com RMSE, MAE e RMSE/MAE no texto."""
    y, yhat, split = _ler_preds(c.preds)
    e = y - yhat
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    xlo, xhi = _janela(yhat)
    elo, ehi = _janela(e)
    dentro = (yhat >= xlo) & (yhat <= xhi) & (e >= elo) & (e <= ehi)
    fora = int((~dentro).sum())
    fig, ax = _novo_fig()
    hb = ax.hexbin(yhat[dentro], e[dentro], gridsize=70, extent=(xlo, xhi, elo, ehi),
                   bins="log", mincnt=1, cmap="viridis", linewidths=0)
    ax.axhline(0, color="red", ls="--", lw=1.2, label="resíduo zero (acerto perfeito)")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(elo, ehi)
    _sci_axis(ax)
    ax.set_xlabel("Previsto ŷ (log-retorno, adimensional)")
    ax.set_ylabel("Resíduo e = y − ŷ (log-retorno, adimensional)")
    ax.set_title(f"Resíduos × previsto {SPLIT_TXT.get(split, split)}")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("pontos por célula (escala log)")
    texto = (f"RMSE = {sci_fmt(rmse)}\n"
             f"MAE = {sci_fmt(mae)}\n"
             f"RMSE/MAE = {_num(round(rmse / mae, 2))} (≈1,25 se gaussiano)\n"
             f"fora do recorte: {_pct(fora / len(y))}")
    ax.text(0.02, 0.97, texto, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8)
    return _salvar(fig, c.out, "fig5_residuos.png")


# ---------------------------------------------------------------- fig6
def fig6_subgrupos(c):
    """RMSE por subgrupo (barras horizontais), com a linha do teste inteiro se a linha 'geral' existir."""
    cam = c.res / f"subgroups_{c.subgroups}.csv"
    if not cam.exists():
        raise Faltando(f"falta {cam} (rode error_analysis.py)")
    df = pd.read_csv(cam)
    exigidas = ["dimensao", "faixa", "n", "rmse"]
    if any(k not in df.columns for k in exigidas):
        raise ValueError(f"{cam.name} sem colunas {exigidas}")
    geral = df[df["dimensao"] == "geral"]
    sub = df[df["dimensao"] != "geral"].reset_index(drop=True)
    dims = list(dict.fromkeys(sub["dimensao"]))
    cores = {d: plt.cm.tab10(i % 10) for i, d in enumerate(dims)}
    fig, ax = _novo_fig()
    rot = [f"{r.dimensao}: {r.faixa} (n = {_inteiro(r.n)})" for r in sub.itertuples()]
    barras = ax.barh(range(len(sub)), sub["rmse"], color=[cores[d] for d in sub["dimensao"]])
    for b, v in zip(barras, sub["rmse"]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + sci_fmt(v),
                va="center", fontsize=7)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(rot, fontsize=7)
    ax.invert_yaxis()
    _sci_axis(ax, y=False)
    ax.set_xlabel("RMSE (log-retorno, adimensional)")
    ax.set_title(f"RMSE por subgrupo {SPLIT_TXT.get(c.subgroups, c.subgroups)}")
    alças = [Patch(color=cores[d], label=d) for d in dims]
    if len(geral):
        g = float(geral["rmse"].iloc[0])
        ax.axvline(g, color="red", ls="--", lw=1.2)
        alças.append(Line2D([0], [0], color="red", ls="--", label=f"teste inteiro (RMSE = {sci_fmt(g)})"))
    ax.legend(handles=alças, fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _salvar(fig, c.out, "fig6_subgrupos.png")


# ---------------------------------------------------------------- fig7
ROTULO_FAMILIA = {
    "ret_1": "retorno ret_1",
    "amplitude": "amplitude",
    "direcao": "direção",
    "volume": "volume (log1p)",
    "rv_60": "volatilidade rv_60",
}


def fig7_importancia(c):
    """ΔRMSE de permutação por família de colunas, na validação, com desvio das repetições."""
    cam = c.res / "importance_val.csv"
    if not cam.exists():
        raise Faltando(f"falta {cam} (rode importance.py)")
    df = pd.read_csv(cam)
    # colunas do importance.py (2b): delta_rmse_media, delta_rmse_dp
    exigidas = ["modelo", "familia", "delta_rmse_media", "delta_rmse_dp"]
    if any(k not in df.columns for k in exigidas):
        raise ValueError(f"{cam.name} sem colunas {exigidas}")
    modelos = list(dict.fromkeys(df["modelo"]))
    primeiro = df[df["modelo"] == modelos[0]].sort_values("delta_rmse_media", ascending=True)
    familias = list(primeiro["familia"])
    pos = np.arange(len(familias))
    larg = 0.8 / len(modelos)
    fig, ax = _novo_fig()
    for i, m in enumerate(modelos):
        sub = df[df["modelo"] == m].set_index("familia").reindex(familias)
        ax.barh(pos + (i - (len(modelos) - 1) / 2) * larg, sub["delta_rmse_media"],
                xerr=sub["delta_rmse_dp"], height=larg, capsize=3, label=m)
    ax.set_yticks(pos)
    ax.set_yticklabels([ROTULO_FAMILIA.get(f, f) for f in familias])
    ax.axvline(0, color="black", lw=0.8)
    _sci_axis(ax, y=False)
    ax.set_xlabel("ΔRMSE após embaralhar a família (log-retorno; maior = mais importante)")
    ax.set_title("Importância por permutação na validação (5 repetições)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="modelo", fontsize=8)
    return _salvar(fig, c.out, "fig7_importancia.png")


FIGURAS = {
    "fig2": fig2_curvas,
    "fig3": fig3_regularizacao,
    "fig4": fig4_previsto_real,
    "fig5": fig5_residuos,
    "fig6": fig6_subgrupos,
    "fig7": fig7_importancia,
}


def _normalizar(nome):
    """Aceita 'fig4', 'fig4_previsto_real' ou '4'."""
    n = nome.lower().split("_")[0]
    if n.isdigit():
        n = "fig" + n
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gera as figuras 2 a 7 do projeto de regressão.")
    ap.add_argument("--only", nargs="+", metavar="FIG", help="ex.: --only fig4 (ou --only fig4 fig5)")
    ap.add_argument("--res", type=Path, default=RES, help="pasta de results (runs/, subgroups_*.csv, importance_val.csv)")
    ap.add_argument("--cache", type=Path, default=CACHE, help="pasta de cache (preds/test_preds.npz)")
    ap.add_argument("--preds", type=Path, default=None, help="npz de previsões para fig4 e fig5 (padrão: cache/preds/test_preds.npz)")
    ap.add_argument("--subgroups", choices=["test", "val"], default="test", help="split do subgroups_<split>.csv da fig6")
    ap.add_argument("--out", type=Path, default=FIG, help="pasta de saída (padrão: figures/)")
    args = ap.parse_args(argv)

    alternativo = args.res != RES or args.cache != CACHE or args.preds is not None
    if alternativo and args.out == FIG:
        ap.error("entradas alternativas (--res/--cache/--preds) exigem --out fora de figures/")

    alvo = list(FIGURAS)
    if args.only:
        alvo = []
        for nome in args.only:
            n = _normalizar(nome)
            if n not in FIGURAS:
                ap.error(f"figura desconhecida: {nome} (use fig2 a fig7)")
            alvo.append(n)

    c = argparse.Namespace(
        res=args.res, cache=args.cache, out=args.out, subgroups=args.subgroups,
        preds=args.preds if args.preds else args.cache / "preds" / "test_preds.npz",
    )
    print(f"entrada: {c.res} | cache: {c.cache} | preds: {c.preds} | saída: {c.out}")
    feitas, faltando, falhas = [], [], []
    for nome in alvo:
        try:
            caminho = FIGURAS[nome](c)
            print(f"ok    {nome}: {caminho}")
            feitas.append(nome)
        except Faltando as e:
            print(f"pulo  {nome}: {e}")
            faltando.append(nome)
        except Exception as e:  # erro de dado não impede as outras figuras
            print(f"ERRO  {nome}: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            falhas.append(nome)
    print(f"resumo: {len(feitas)} gerada(s), {len(faltando)} pulada(s) por falta de dado, {len(falhas)} com erro")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
