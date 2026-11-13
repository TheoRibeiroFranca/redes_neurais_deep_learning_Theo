# Busca em grade da entrega 2 (regressão do retorno de 1 min, BTC/USD).
# Seção 4A: 4 combinações de regularização. Aula 7: weight decay (AdamW) e dropout.
# Seção 4B: 12 configurações (g01 a g12) em 5 etapas, cada uma a "melhor até aí".
#   Aula 6: lr em grade logarítmica (3e-4, 1e-3, 3e-3); batch 4096 (regra da raiz, com milhões de linhas).
#   Aula 8.2: Huber para caudas pesadas; MSE como comparação.
# Aula 2 (Splitting §4): o melhor de k candidatos é otimista, então as 16 candidatas são gravadas.
# A escolha é sempre por val_rmse. O teste não é lido aqui.
# Retomável: um run cujo json já existe em results/runs é lido, não refeito.
# Uso: python search.py [--smoke] [--workers 3] [--threads 4]

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE = Path(__file__).resolve().parent
RES = CODE.parent / "results"
RUNS = RES / "runs"
SEED = 42

# Base da 4A: F65, janela completa, [128, 64], lr 1e-3, Huber.
BASE = {
    "feature_set": "F65", "window": "full", "hidden": [128, 64],
    "dropout": 0.0, "weight_decay": 0.0, "lr": 1e-3, "batch": 4096,
    "max_epochs": 30, "patience": 5, "loss": "huber", "huber_delta": 1.0,
    "seed": SEED, "max_rows": None,
}
SMOKE_OVERRIDES = {"max_rows": 100000, "max_epochs": 1}

# Seção 4A: nome, weight_decay, dropout.
REG_4A = [
    ("reg_none", 0.0, 0.0),
    ("reg_wd", 0.01, 0.0),
    ("reg_dropout", 0.0, 0.2),
    ("reg_wd_dropout", 0.01, 0.2),
]

COLS = ["name", "feature_set", "window", "hidden", "lr", "loss", "weight_decay", "dropout",
        "n_params", "best_epoch", "val_rmse", "val_mae", "val_r2"]


def score(rec):
    """val_rmse para comparar. NaN (run divergido) perde sempre."""
    v = rec["val"]["rmse"]
    return math.inf if math.isnan(v) else v


def best_name(names, done):
    """Menor val_rmse entre os nomes. Empate: o primeiro da lista. Sem run válida, para."""
    winner = min(names, key=lambda n: score(done[n][1]))
    if score(done[winner][1]) == math.inf:
        raise RuntimeError(f"nenhuma run válida entre {names}; veja as falhas acima")
    return winner


def failed_record(cfg, err):
    """Registro só em memória para um run que levantou erro. Não grava json: na próxima
    execução o run é tentado de novo (ex.: divergência logo na 1ª época, ou cache faltando)."""
    nan = float("nan")
    return {"cfg": cfg, "n_params": None, "best_epoch": None,
            "val": {"rmse": nan, "mae": nan, "r2": nan}, "failed": repr(err)}


def run_stage(cfgs, train_fn, workers, runs_dir):
    """Roda as cfgs em paralelo (ProcessPoolExecutor). Quem já tem json é lido e não refeito."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for cfg in cfgs:
        path = runs_dir / f"{cfg['name']}.json"
        if path.exists():
            out[cfg["name"]] = json.loads(path.read_text(encoding="utf-8"))
            print(f"[pula] {cfg['name']}: json existe", flush=True)
        else:
            todo.append(cfg)
    if todo:
        cfg_of = {c["name"]: c for c in todo}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(train_fn, cfg): cfg["name"] for cfg in todo}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    out[name] = fut.result()
                    print(f"[ok] {name}: val_rmse={out[name]['val']['rmse']:.6g}", flush=True)
                except Exception as err:  # falha de um run não derruba a etapa
                    out[name] = failed_record(cfg_of[name], err)
                    print(f"[falhou] {name}: {err!r} (sem json; rode de novo para tentar)", flush=True)
    return out


def row(cfg, rec):
    v = rec["val"]
    return {
        "name": cfg["name"], "feature_set": cfg["feature_set"], "window": cfg["window"],
        "hidden": "-".join(str(h) for h in cfg["hidden"]), "lr": cfg["lr"], "loss": cfg["loss"],
        "weight_decay": cfg["weight_decay"], "dropout": cfg["dropout"],
        "n_params": rec["n_params"], "best_epoch": rec["best_epoch"],
        "val_rmse": v["rmse"], "val_mae": v["mae"], "val_r2": v["r2"],
    }


def write_grid(path, rows):
    rows = sorted(rows, key=lambda r: math.inf if math.isnan(r["val_rmse"]) else r["val_rmse"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def search(train_fn, smoke, workers, threads):
    tag = "_smoke" if smoke else ""
    pre = "smoke_" if smoke else ""
    base = {**BASE, "threads": threads, **(SMOKE_OVERRIDES if smoke else {})}
    n4a = [pre + n for n, _, _ in REG_4A]
    n4b = [pre + f"g{i:02d}" for i in range(1, 13)]
    done = {}  # nome -> (cfg, record)
    RES.mkdir(parents=True, exist_ok=True)

    def stage(cfgs):
        recs = run_stage(cfgs, train_fn, workers, RUNS)
        for c in cfgs:
            done[c["name"]] = (c, recs[c["name"]])

    def save_4a():
        write_grid(RES / f"grid_4a{tag}.csv", [row(*done[n]) for n in n4a if n in done])

    def save_4b():
        write_grid(RES / f"grid_4b{tag}.csv", [row(*done[n]) for n in n4b if n in done])

    def derive(src_names, new_name, **changes):
        """Cópia da melhor (até aí) entre src_names, com nome novo e mudanças."""
        src = done[best_name(src_names, done)][0]
        return {**src, "name": pre + new_name, **changes}

    # 4A: regularização, base F65 full [128, 64].
    cfgs = [{**base, "name": pre + n, "weight_decay": wd, "dropout": dp} for n, wd, dp in REG_4A]
    stage(cfgs)
    save_4a()
    reg = done[best_name(n4a, done)][0]
    print(f"4A: melhor regularização = {reg['name']} (weight_decay={reg['weight_decay']}, dropout={reg['dropout']})", flush=True)

    # g01-g08: F4/F65 x full/2018 x [64, 32]/[256, 128], lr 1e-3, Huber, regularização de 4A.
    cfgs = []
    for fs in ("F4", "F65"):
        for win in ("full", "2018"):
            for hid in ([64, 32], [256, 128]):
                cfgs.append({**base, "name": pre + f"g{len(cfgs) + 1:02d}", "feature_set": fs,
                             "window": win, "hidden": hid, "weight_decay": reg["weight_decay"],
                             "dropout": reg["dropout"], "lr": 1e-3, "loss": "huber"})
    stage(cfgs)
    save_4b()

    # g09, g10: melhor de g01-g08 com lr 3e-4 e 3e-3.
    cfgs = [derive(n4b[:8], "g09", lr=3e-4), derive(n4b[:8], "g10", lr=3e-3)]
    stage(cfgs)
    save_4b()

    # g11: melhor de g01-g10 com [256, 128, 64].
    stage([derive(n4b[:10], "g11", hidden=[256, 128, 64])])
    save_4b()

    # g12: melhor de g01-g11 com loss mse.
    stage([derive(n4b[:11], "g12", loss="mse")])
    save_4b()

    all_names = n4a + n4b
    winner = best_name(all_names, done)
    print(f"Configurações tentadas: {len(n4a)} em 4A + {len(n4b)} em 4B = {len(all_names)}", flush=True)
    print(f"Melhor por val_rmse: {winner} (val_rmse={done[winner][1]['val']['rmse']:.6g})", flush=True)
    print(f"Saídas: {RES / ('grid_4a' + tag + '.csv')} e {RES / ('grid_4b' + tag + '.csv')}", flush=True)


def main(argv=None, train_fn=None):
    ap = argparse.ArgumentParser(description="Busca em grade: seção 4A e g01-g12 da 4B.")
    ap.add_argument("--smoke", action="store_true", help="max_rows=100000, 1 época; grava nomes smoke_*")
    ap.add_argument("--workers", type=int, default=3, help="processos em paralelo")
    ap.add_argument("--threads", type=int, default=4, help="threads de torch por run")
    args = ap.parse_args(argv)
    if train_fn is None:
        from train import train_run as train_fn
    search(train_fn, args.smoke, args.workers, args.threads)


if __name__ == "__main__":
    main()
