# Avaliação final: rede média de 3 sementes e baselines, uma vez só, no teste.
# Aula 2 (Splitting, "The golden rule"): o teste é aberto uma vez; a trava recusa nova execução.
# Aula 2 (Scaling the target): a rede prevê y / y_std; a saída volta à escala original antes de medir.
# Aula 8.2 (Regression): MAE, RMSE e R2 na escala original; persistência = último valor observado.
# Aula 7 (Forgetting model.eval): model.eval() e no_grad na inferência.
# Aula 2 (Splitting, Lab Step 1): média ± desvio entre sementes; critério com 0,5% e 2 desvios.
# --dry-run-val repete tudo na validação e grava com sufixo _dryrun_val (o teste não é lido).
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import pickle

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from features import load_arrays
from model import MLP


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RES = ROOT / "docs" / "projects" / "regression" / "results"
CACHE = ROOT / "data" / "cache"

SEEDS = (42, 43, 44)
BATCH = 65536
FATOR_CRITERIO = 0.995  # rede precisa estar até 0,5% abaixo do melhor baseline
K_DESVIOS = 2           # e a diferença precisa passar de 2 desvios-padrão das sementes


def checar_trava(dry_run: bool) -> None:
    """Recusa rodar se o teste já foi aberto (Aula 2, golden rule)."""
    marcas = [RES / "test_opened.json", RES / "test_metrics.json"]
    if not dry_run:
        marcas.append(CACHE / "preds" / "test_preds.npz")
    for marca in marcas:
        if marca.exists():
            raise SystemExit(
                f"Recusado: {marca} já existe. O teste é aberto uma vez só. "
                "Para testar o script, use --dry-run-val."
            )


def marcar_teste_aberto() -> None:
    """Grava o marcador antes de ler X_te/y_te: se o script cair depois, o uso fica registrado."""
    RES.mkdir(parents=True, exist_ok=True)
    marca = {"timestamp": datetime.now().isoformat(timespec="seconds"), "script": "evaluate_test.py"}
    with open(RES / "test_opened.json", "x", encoding="utf-8") as f:
        json.dump(marca, f, indent=2)


def dados_do_split(arrays: dict, parte: str) -> dict:
    return {
        "X": arrays[f"X_{parte}"],
        "y": arrays[f"y_{parte}"],
        "ret1": arrays[f"ret1_{parte}"],
        "ts": arrays[f"ts_{parte}"],
        "vol0": arrays[f"vol0_{parte}"],
        "rv60": arrays[f"rv60_{parte}"],
    }


def preparar(fs: str, window: str, parte: str) -> dict:
    """Carrega uma combinação F4/F65 x full/2018 e guarda só o que o script usa."""
    arr = load_arrays(fs, window)
    pacote = {
        "dados": dados_do_split(arr, parte),
        "media_tr": float(arr["y_tr"].mean()),
        "mediana_tr": float(np.median(arr["y_tr"])),
        "y_std": float(arr["y_std"]),
    }
    del arr
    return pacote


def carregar_rede(cfg: dict, n_in: int, nome: str) -> MLP:
    rede = MLP(n_in, list(cfg["hidden"]), dropout=float(cfg["dropout"]))
    estado = torch.load(RES / "models" / f"{nome}.pt", map_location="cpu", weights_only=True)
    rede.load_state_dict(estado)
    rede.eval()
    return rede


@torch.no_grad()
def prever_rede(rede: MLP, X: np.ndarray, y_std: float) -> np.ndarray:
    saida = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), BATCH):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH], dtype=np.float32))
        saida[i:i + BATCH] = rede(xb).squeeze(-1).numpy()
    return saida * y_std  # volta à escala original


def prever_baseline(modelo: str, pacote: dict, fs: str, window: str) -> np.ndarray:
    n = len(pacote["dados"]["y"])
    if modelo == "dummy_mean":
        return np.full(n, pacote["media_tr"])
    if modelo == "dummy_median":
        return np.full(n, pacote["mediana_tr"])
    if modelo == "zero":
        return np.zeros(n)
    if modelo == "persistencia":  # previsão de ret_1 do minuto t (último valor observado)
        return pacote["dados"]["ret1"].astype(np.float64)
    if modelo == "ridge":
        with open(RES / "models" / f"ridge_{fs}_{window}.pkl", "rb") as f:
            ridge = pickle.load(f)
        return ridge.predict(pacote["dados"]["X"])
    raise ValueError(f"baseline desconhecido em baselines_best.json: {modelo}")


def metricas(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "rmse": float(root_mean_squared_error(y, yhat)),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação final no teste (uma vez) ou dry-run na validação.")
    ap.add_argument("--dry-run-val", action="store_true",
                    help="repete tudo na validação e grava com sufixo _dryrun_val")
    args = ap.parse_args()
    dry = args.dry_run_val
    checar_trava(dry)
    torch.set_num_threads(int(os.environ.get("NUM_THREADS", 3)))

    parte = "val" if dry else "te"
    sufixo = "_dryrun_val" if dry else ""
    nome_split = "val" if dry else "test"

    chosen = json.loads((RES / "chosen.json").read_text(encoding="utf-8"))
    cfg = chosen["config"]
    fs, window = cfg["feature_set"], cfg["window"]
    melhor = json.loads((RES / "baselines_best.json").read_text(encoding="utf-8"))["melhor"]
    base_fs, base_window = melhor["feature_set"], melhor["window"]

    if not dry:
        marcar_teste_aberto()  # antes de qualquer leitura de X_te/y_te

    # Rede: três sementes, previsões na escala original.
    p_net = preparar(fs, window, parte)
    d = p_net["dados"]
    n_in = d["X"].shape[1]
    previsoes = []
    for s in SEEDS:
        rede = carregar_rede(cfg, n_in, f"final_s{s}")
        previsoes.append(prever_rede(rede, d["X"], p_net["y_std"]))
        print(f"semente {s}: previsto")
    yhat_seeds = np.column_stack(previsoes)
    yhat_media = yhat_seeds.mean(axis=1)

    por_semente = [{"seed": s, **metricas(d["y"], yhat_seeds[:, i])} for i, s in enumerate(SEEDS)]
    rede_media = {k: float(np.mean([m[k] for m in por_semente])) for k in ("rmse", "mae", "r2")}
    rede_dp = {k: float(np.std([m[k] for m in por_semente], ddof=1)) for k in ("rmse", "mae", "r2")}
    rede_ensemble = metricas(d["y"], yhat_media)

    # Baselines no mesmo split. O melhor vem do json escolhido na validação.
    base_pred = {
        "media_treino": prever_baseline("dummy_mean", p_net, fs, window),
        "zero": prever_baseline("zero", p_net, fs, window),
        "persistencia": prever_baseline("persistencia", p_net, fs, window),
        "ridge_mesmo_conjunto": prever_baseline("ridge", p_net, fs, window),
    }
    base_metricas = {nome: metricas(d["y"], v) for nome, v in base_pred.items()}

    p_base = p_net if (base_fs, base_window) == (fs, window) else preparar(base_fs, base_window, parte)
    db = p_base["dados"]
    if not (np.array_equal(db["y"], d["y"]) and np.array_equal(db["ts"], d["ts"])):
        raise SystemExit("As linhas do melhor baseline não batem com as da rede: alvo ou ts diferentes.")
    melhor_pred = prever_baseline(melhor["model"], p_base, base_fs, base_window)
    melhor_metricas = metricas(db["y"], melhor_pred)

    # Critério: duas condições, gravadas separadamente.
    rmse_rede = rede_media["rmse"]
    rmse_melhor = melhor_metricas["rmse"]
    limite_cond1 = FATOR_CRITERIO * rmse_melhor
    diferenca = rmse_melhor - rmse_rede
    limite_cond2 = K_DESVIOS * rede_dp["rmse"]
    cond1 = rmse_rede <= limite_cond1
    cond2 = diferenca > limite_cond2
    aprovado = bool(cond1 and cond2)

    RES.mkdir(parents=True, exist_ok=True)
    (CACHE / "preds").mkdir(parents=True, exist_ok=True)

    linhas = [("rede_semente_%d" % m["seed"], m["rmse"], m["mae"], m["r2"]) for m in por_semente]
    linhas.append(("rede_media_3_sementes", rede_media["rmse"], rede_media["mae"], rede_media["r2"]))
    linhas.append(("rede_desvio_3_sementes", rede_dp["rmse"], rede_dp["mae"], rede_dp["r2"]))
    linhas.append(("rede_yhat_media_sementes", rede_ensemble["rmse"], rede_ensemble["mae"], rede_ensemble["r2"]))
    for nome, m in base_metricas.items():
        linhas.append((f"baseline_{nome}", m["rmse"], m["mae"], m["r2"]))
    linhas.append((f"melhor_baseline_validacao_{melhor['model']}", melhor_metricas["rmse"],
                   melhor_metricas["mae"], melhor_metricas["r2"]))

    with open(RES / f"test_table{sufixo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "rmse", "mae", "r2"])
        for nome, rmse, mae, r2 in linhas:
            w.writerow([nome, f"{rmse:.8f}", f"{mae:.8f}", f"{r2:.8f}"])

    np.savez(
        CACHE / "preds" / f"test_preds{sufixo}.npz",
        y=d["y"], yhat=yhat_media, yhat_seeds=yhat_seeds,
        ts=d["ts"], vol0=d["vol0"], rv60=d["rv60"], ret1=d["ret1"],
        split=nome_split,
    )

    resultado = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modo": "dryrun_val" if dry else "teste",
        "split": nome_split,
        "n_linhas": int(len(d["y"])),
        "chosen": cfg,
        "por_semente": por_semente,
        "rede_media_3_sementes": rede_media,
        "rede_desvio_3_sementes": rede_dp,
        "rede_yhat_media_sementes": rede_ensemble,
        "baselines": base_metricas,
        "melhor_baseline": {
            "modelo": melhor["model"],
            "feature_set": base_fs,
            "window": base_window,
            "rmse_no_json_validacao": melhor.get("rmse"),
            **melhor_metricas,
        },
        "criterio": {
            "regra": "rmse_rede_media <= 0,995 x rmse_melhor_baseline E diferenca > 2 x dp_rmse_sementes",
            "rmse_rede_media": rmse_rede,
            "rmse_melhor_baseline": rmse_melhor,
            "cond1_limite": limite_cond1,
            "cond1_ok": bool(cond1),
            "diferenca": diferenca,
            "cond2_limite": limite_cond2,
            "cond2_ok": bool(cond2),
        },
        "aprovado": aprovado,
    }
    (RES / f"test_metrics{sufixo}.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSplit: {nome_split} ({len(d['y'])} linhas)")
    print(f"Rede media (3 sementes): RMSE {rede_media['rmse']:.8f} +- {rede_dp['rmse']:.8f}")
    print(f"Melhor baseline ({melhor['model']}): RMSE {rmse_melhor:.8f}")
    print(f"Cond. 1 (rede <= 0,995 x baseline): {cond1}")
    print(f"Cond. 2 (diferenca {diferenca:.8f} > 2 x dp {limite_cond2:.8f}): {cond2}")
    print(f"Aprovado: {aprovado}")
    print(f"Gravado: {RES / ('test_metrics' + sufixo + '.json')}")


if __name__ == "__main__":
    main()
