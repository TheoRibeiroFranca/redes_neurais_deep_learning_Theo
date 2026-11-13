# Análise de erros da regressão (entrega 2, BTC/USD 1 min).
# Erro por subgrupo, 10 piores casos e acerto do sinal (extra; sem backtest, custo ou P&L).
# Aula 8.2 (Regression): "Report error by segment" (funil, heterocedasticidade);
#   "One outlier" (piores casos); RMSE/MAE como alarme (razão RMSE/MAE).
# Aula 8.1 (Classification): matriz de confusão e acurácia, comparadas com o baseline trivial.
# Aula 2 (Splitting): os limites de rv60 vêm só do treino, nunca de validação ou teste.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RES = Path(__file__).resolve().parents[1] / "results"
CACHE = ROOT / "data" / "cache"
PREDS_PADRAO = CACHE / "preds" / "test_preds.npz"
F65_PADRAO = CACHE / "F65_full.npz"
SPLITS = ("val", "test")
CHAVES = ("y", "yhat", "ts", "vol0", "rv60", "ret1", "split")
N_PIORES = 10


def ler_preds(caminho: Path):
    with np.load(caminho, allow_pickle=False) as d:
        faltando = [k for k in CHAVES if k not in d.files]
        if faltando:
            raise SystemExit(f"{caminho}: faltam as chaves {faltando}")
        split = str(np.asarray(d["split"]).item())
        dados = {k: np.asarray(d[k]) for k in CHAVES if k != "split"}
    if split not in SPLITS:
        raise SystemExit(f"{caminho}: split '{split}' inválido (esperado 'val' ou 'test')")
    tamanhos = {k: v.shape[0] for k, v in dados.items()}
    if len(set(tamanhos.values())) != 1:
        raise SystemExit(f"{caminho}: chaves com tamanhos diferentes {tamanhos}")
    out = {
        "y": dados["y"].astype(np.float64),
        "yhat": dados["yhat"].astype(np.float64),
        "ts": dados["ts"].astype(np.int64),
        "vol0": dados["vol0"].astype(bool),
        "rv60": dados["rv60"].astype(np.float64),
        "ret1": dados["ret1"].astype(np.float64),
    }
    if not (np.isfinite(out["y"]).all() and np.isfinite(out["yhat"]).all()):
        raise SystemExit(f"{caminho}: y ou yhat com NaN/inf")
    return out, split


def tercis_rv60(arg_cuts, caminho_f65: Path):
    # --rv60-cuts, se dado, tem prioridade; senão tercis de rv60_tr do cache de treino.
    if arg_cuts:
        partes = arg_cuts.split(",")
        if len(partes) != 2:
            raise SystemExit("--rv60-cuts espera 'a,b'")
        a, b = float(partes[0]), float(partes[1])
        origem = "--rv60-cuts"
    else:
        if not caminho_f65.exists():
            raise SystemExit(f"cache {caminho_f65} não existe; passe --rv60-cuts a,b")
        with np.load(caminho_f65) as d:
            rv_tr = np.asarray(d["rv60_tr"], dtype=np.float64)
        rv_tr = rv_tr[np.isfinite(rv_tr)]
        a, b = (float(v) for v in np.quantile(rv_tr, [1 / 3, 2 / 3]))
        origem = f"tercis de rv60_tr do treino ({rv_tr.size} linhas, {caminho_f65.name})"
    if not a <= b:
        raise SystemExit(f"limites de rv60 fora de ordem: {a}, {b}")
    return a, b, origem


def resumo(dimensao, grupos, erro):
    linhas = []
    for rotulo, mascara in grupos:
        e = erro[mascara]
        n = int(e.size)
        if n == 0:
            mae = rmse = razao = np.nan
        else:
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e ** 2)))
            razao = rmse / mae if mae > 0 else np.nan
        linhas.append({"dimensao": dimensao, "faixa": rotulo, "n": n,
                       "mae": mae, "rmse": rmse, "rmse_mae": razao})
    return linhas


def subgrupos(d, cuts):
    erro = d["y"] - d["yhat"]
    absy = np.abs(d["y"])
    dt = pd.to_datetime(d["ts"], unit="s", utc=True)
    a, b = cuts
    q = np.quantile(absy, [0.2, 0.4, 0.6, 0.8])
    q99 = float(np.quantile(absy, 0.99))
    quintil = np.searchsorted(q, absy, side="right")  # 0..4
    lim = [float(absy.min()), *map(float, q), float(absy.max())]
    rv = d["rv60"]
    horas = np.asarray(dt.hour) // 6
    blocos = ["00-05h", "06-11h", "12-17h", "18-23h"]

    linhas = []
    linhas += resumo("volume", [("Volume = 0", d["vol0"]),
                                ("com negócio (Volume > 0)", ~d["vol0"])], erro)
    linhas += resumo("|y| quintil", [
        (f"Q{k + 1}: |y| de {lim[k]:.3e} a {lim[k + 1]:.3e}", quintil == k) for k in range(5)], erro)
    linhas += resumo("|y| top 1%", [
        (f"top 1% (|y| >= {q99:.3e})", absy >= q99),
        (f"demais 99% (|y| < {q99:.3e})", absy < q99)], erro)
    linhas += resumo("rv60 tercil", [
        (f"T1: rv60 <= {a:.3e}", rv <= a),
        (f"T2: {a:.3e} < rv60 <= {b:.3e}", (rv > a) & (rv <= b)),
        (f"T3: rv60 > {b:.3e}", rv > b)], erro)
    linhas += resumo("ano (UTC)", [(str(ano), np.asarray(dt.year) == ano)
                                   for ano in sorted(set(np.asarray(dt.year)))], erro)
    linhas += resumo("hora UTC", [(blocos[k], horas == k) for k in range(4)], erro)
    return linhas


def piores(d):
    erro_abs = np.abs(d["y"] - d["yhat"])
    idx = np.argsort(-erro_abs, kind="stable")[:N_PIORES]
    ts_utc = pd.to_datetime(d["ts"][idx], unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    return pd.DataFrame({
        "posicao": np.arange(1, idx.size + 1),
        "ts_utc": np.asarray(ts_utc),
        "y": d["y"][idx],
        "yhat": d["yhat"][idx],
        "erro_abs": erro_abs[idx],
        "vol0": d["vol0"][idx],
        "rv60": d["rv60"][idx],
        "ret1": d["ret1"][idx],
    })


def sinal(d):
    # Extra de sinal: só y != 0. Acerto conta ŷ = 0 como erro; a matriz exclui ŷ = 0 (n_yhat_zero).
    y, yh = d["y"], d["yhat"]
    m = y != 0
    yy, yhm = y[m], yh[m]
    real_sobe = yy > 0
    acerto = float(np.mean(np.sign(yhm) == np.sign(yy)))
    tem = yhm != 0
    rs, ps = real_sobe[tem], yhm[tem] > 0
    base = float(max(real_sobe.mean(), 1 - real_sobe.mean()))
    return {
        "n_linhas": int(y.size),
        "n_y_diferente_de_zero": int(yy.size),
        "n_yhat_zero": int(np.sum(yhm == 0)),
        "acerto": acerto,
        "acerto_baseline_classe_majoritaria": base,
        "matriz": {
            "real_sobe": {"prev_sobe": int(np.sum(rs & ps)), "prev_desce": int(np.sum(rs & ~ps))},
            "real_desce": {"prev_sobe": int(np.sum(~rs & ps)), "prev_desce": int(np.sum(~rs & ~ps))},
        },
        "nota": "extra de sinal: sem backtest, custo ou P&L; yhat = 0 conta como erro no acerto",
    }


def main():
    ap = argparse.ArgumentParser(description="Análise de erros da regressão (subgrupos, piores casos, sinal).")
    ap.add_argument("--preds", type=Path, default=PREDS_PADRAO)
    ap.add_argument("--out", type=Path, default=RES, help="pasta de saída (padrão: results)")
    ap.add_argument("--f65-cache", type=Path, default=F65_PADRAO)
    ap.add_argument("--rv60-cuts", type=str, default=None, help="limites 'a,b' de rv60; sobrepõe o cache")
    args = ap.parse_args()

    d, split = ler_preds(args.preds)
    a, b, origem = tercis_rv60(args.rv60_cuts, args.f65_cache)
    args.out.mkdir(parents=True, exist_ok=True)

    tab = pd.DataFrame(subgrupos(d, (a, b)))
    tab.to_csv(args.out / f"subgroups_{split}.csv", index=False, float_format="%.8g")

    pior = piores(d)
    pior.to_csv(args.out / f"worst10_{split}.csv", index=False, float_format="%.8g")

    sg = sinal(d)
    with open(args.out / f"sign_{split}.json", "w", encoding="utf-8") as f:
        json.dump(sg, f, indent=2, ensure_ascii=False)

    periodo = pd.to_datetime(d["ts"], unit="s", utc=True)
    print(f"split={split} n={d['y'].size} periodo={periodo.min()} .. {periodo.max()}")
    print(f"preds={args.preds}")
    print(f"rv60 tercis: a={a:.6g} b={b:.6g} ({origem})")
    print(f"saídas em {args.out}")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n10 piores:")
    print(pior.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nsinal:", json.dumps(sg, ensure_ascii=False))


if __name__ == "__main__":
    main()
