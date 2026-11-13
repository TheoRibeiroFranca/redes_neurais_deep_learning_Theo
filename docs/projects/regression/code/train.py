# Treino de uma configuração da MLP (passos 2 e 3) com early stopping pela validação.
# Loop: esqueleto do enunciado (early stopping, paciência, restaurar o melhor state_dict).
# Aula 6 (Otimização): AdamW, lr 1e-3 como ponto de partida; weight decay só nos pesos,
#   vieses sem decay ("Do not decay every parameter"); mini-batch embaralhado a cada época.
# Aula 7 (Regularização): early stopping em três partes (validação por época, paciência,
#   restaurar o melhor checkpoint); model.eval() na avaliação por causa do dropout.
# Aula 8.2 (Métricas de regressão): perda Huber (quadrática até delta, linear depois) ou MSE;
#   RMSE, MAE e R² como métricas.
# Aula 2 (Scaling the target): a perda usa y / y_std (y_std do treino); as métricas voltam
#   à escala original antes de medir.
# Handout de Otimização: §5.1 perda no conjunto inteiro por época (treino e validação, em
#   lotes, sem gradiente); tensores criados uma vez, sem DataLoader; semente 42.
# O teste não é lido aqui: X_te/y_te são descartados logo depois de carregar.
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from features import load_arrays
from model import MLP, n_params

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RES = CODE.parent / "results"
RUNS_DIR = RES / "runs"
MODELS_DIR = RES / "models"
PREDS_DIR = ROOT / "data" / "cache" / "preds"

EVAL_BS = 65_536

DEFAULTS = {
    "name": None,
    "feature_set": "F65",
    "window": "full",
    "hidden": [128, 64],
    "dropout": 0.0,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "batch": 4096,
    "max_epochs": 30,
    "patience": 5,
    "loss": "huber",
    "huber_delta": 1.0,
    "seed": 42,
    "threads": int(os.environ.get("NUM_THREADS", 3)),
    "max_rows": None,
}


def make_loss(cfg, reduction="mean"):
    if cfg["loss"] == "huber":
        return nn.HuberLoss(delta=cfg["huber_delta"], reduction=reduction)
    if cfg["loss"] == "mse":
        return nn.MSELoss(reduction=reduction)
    raise ValueError(f"loss desconhecida: {cfg['loss']}")


def make_optimizer(model, cfg):
    # Aula 6: dois grupos, pesos (matrizes) com decay e vieses sem decay
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"],
    )


@torch.no_grad()
def evaluate(model, X, y_s, loss_sum, bs=EVAL_BS):
    """Perda média no conjunto inteiro (y padronizado) e previsões na escala padronizada."""
    model.eval()
    total = 0.0
    preds = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        total += loss_sum(out, y_s[i:i + bs]).item()
        preds[i:i + bs] = out.numpy()
    return total / len(X), preds


def metrics(y, yhat):
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    return {
        "rmse": float(root_mean_squared_error(y, yhat)),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }


def train_run(cfg: dict) -> dict:
    cfg = {**DEFAULTS, **cfg}
    if not cfg["name"]:
        raise ValueError("cfg['name'] é obrigatório")
    cfg["hidden"] = [int(h) for h in cfg["hidden"]]
    name = cfg["name"]
    t_start = time.time()

    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    d = load_arrays(cfg["feature_set"], cfg["window"])
    for k in ("X_te", "y_te"):  # regra 4 do contrato: teste fora do desenvolvimento
        d.pop(k, None)
    y_std = float(d["y_std"])

    X_tr_np, y_tr_np = d["X_tr"], d["y_tr"]
    if cfg["max_rows"] is not None and cfg["max_rows"] < len(X_tr_np):
        # subamostra só para smoke test; y_std e escala continuam os do tr completo
        idx = np.sort(rng.choice(len(X_tr_np), size=int(cfg["max_rows"]), replace=False))
        X_tr_np, y_tr_np = X_tr_np[idx], y_tr_np[idx]

    # tensores criados uma vez (from_numpy não copia); o alvo da perda é y / y_std
    X_tr = torch.from_numpy(np.ascontiguousarray(X_tr_np, dtype=np.float32))
    y_tr_s = torch.from_numpy((y_tr_np / y_std).astype(np.float32))
    X_val = torch.from_numpy(np.ascontiguousarray(d["X_val"], dtype=np.float32))
    y_val = d["y_val"].astype(np.float32)
    y_val_s = torch.from_numpy(y_val / y_std)
    n = len(X_tr)

    torch.manual_seed(cfg["seed"])  # mesma inicialização para a mesma semente
    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"])
    opt = make_optimizer(model, cfg)
    loss_fn = make_loss(cfg)
    loss_sum = make_loss(cfg, reduction="sum")
    g = torch.Generator().manual_seed(cfg["seed"])

    for p in (RUNS_DIR, MODELS_DIR, PREDS_DIR):
        p.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / f"{name}.pt"

    history = []
    best, wait, best_epoch, stop_epoch = float("inf"), 0, None, 0
    bs = int(cfg["batch"])
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=g)  # embaralha a cada época
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_tr[idx]), y_tr_s[idx])
            loss.backward()
            opt.step()

        train_loss, _ = evaluate(model, X_tr, y_tr_s, loss_sum)  # treino inteiro, modo eval
        val_loss, pv = evaluate(model, X_val, y_val_s, loss_sum)
        m = metrics(y_val, pv * y_std)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_rmse": m["rmse"], "val_mae": m["mae"],
                        "seconds": time.time() - t0})
        stop_epoch = epoch
        print(f"[{name}] época {epoch:2d}  train {train_loss:.6f}  val {val_loss:.6f}  "
              f"val_rmse {m['rmse']:.4e}  {history[-1]['seconds']:.1f}s", flush=True)

        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            print(f"[{name}] divergiu na época {epoch}", flush=True)
            break
        if val_loss < best:
            best, wait, best_epoch = val_loss, 0, epoch
            torch.save(model.state_dict(), best_path)  # guarda os melhores pesos
        else:
            wait += 1
            if wait == cfg["patience"]:
                break

    if best_epoch is None:
        raise RuntimeError(f"[{name}] nenhuma época com perda de validação finita")
    model.load_state_dict(torch.load(best_path, weights_only=True))  # restaura o melhor

    val_loss_best, pv = evaluate(model, X_val, y_val_s, loss_sum)
    yhat_val = (pv * y_std).astype(np.float32)
    val = metrics(y_val, yhat_val)

    np.savez(PREDS_DIR / f"{name}_val.npz",
             y=y_val, yhat=yhat_val, ts=d["ts_val"], vol0=d["vol0_val"],
             rv60=d["rv60_val"], ret1=d["ret1_val"], split=np.array("val"))

    result = {
        "cfg": cfg,
        "n_params": n_params(model),
        "n_train_rows": int(n),
        "history": history,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "best_val_loss": val_loss_best,
        "y_std": y_std,
        "val": val,
        "seconds_total": time.time() - t_start,
    }
    with open(RUNS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[{name}] melhor época {best_epoch} (parou na {stop_epoch})  "
          f"val rmse {val['rmse']:.6e}  mae {val['mae']:.6e}  r2 {val['r2']:.5f}  "
          f"{result['seconds_total']:.0f}s", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Treina uma MLP com early stopping pela validação.")
    p.add_argument("--name", required=True)
    p.add_argument("--feature-set", default=DEFAULTS["feature_set"], choices=["F4", "F65"])
    p.add_argument("--window", default=DEFAULTS["window"], choices=["full", "2018"])
    p.add_argument("--hidden", type=int, nargs="+", default=DEFAULTS["hidden"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--loss", default=DEFAULTS["loss"], choices=["huber", "mse"])
    p.add_argument("--huber-delta", type=float, default=DEFAULTS["huber_delta"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    p.add_argument("--max-rows", type=int, default=None)
    return vars(p.parse_args())


if __name__ == "__main__":
    train_run(parse_args())
