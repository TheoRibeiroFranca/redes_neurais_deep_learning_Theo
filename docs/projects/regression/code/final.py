# Retreino final da entrega 2: melhor configuração de 4A ∪ 4B por val_rmse, com 3 sementes.
# Aula 7 (Stopping early): o melhor checkpoint é restaurado dentro de train_run.
# Aula 2 (Splitting, Lab Step 1): média ± desvio entre sementes, para comparar com o baseline.
# A escolha usa só a validação. O teste não é lido aqui.
# Uso: python final.py [--workers 3]

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from search import run_stage, score

CODE = Path(__file__).resolve().parent
RES = CODE.parent / "results"
RUNS = RES / "runs"
SEEDS = (42, 43, 44)
N_CANDIDATES = 16  # 4 de 4A + 12 de 4B


def load_candidates():
    """Lê os nomes de grid_4a.csv e grid_4b.csv e o json de cada run."""
    names = []
    for fn in ("grid_4a.csv", "grid_4b.csv"):
        with open(RES / fn, newline="", encoding="utf-8") as f:
            names += [r["name"] for r in csv.DictReader(f)]
    if len(names) != N_CANDIDATES or len(set(names)) != N_CANDIDATES:
        sys.exit(f"esperava {N_CANDIDATES} candidatas únicas em grid_4a/grid_4b, achei {len(names)}")
    recs = {}
    for n in names:
        path = RUNS / f"{n}.json"
        if not path.exists():
            sys.exit(f"falta {path}; rode search.py antes")
        recs[n] = json.loads(path.read_text(encoding="utf-8"))
    return recs


def main(argv=None, train_fn=None):
    ap = argparse.ArgumentParser(description="Retreino final com 3 sementes da melhor por val_rmse.")
    ap.add_argument("--workers", type=int, default=3, help="processos em paralelo")
    args = ap.parse_args(argv)
    if train_fn is None:
        from train import train_run as train_fn

    cands = load_candidates()
    best = min(cands, key=lambda n: score(cands[n]))
    cfg = cands[best]["cfg"]
    print(f"Escolhida por val_rmse entre {N_CANDIDATES}: {best} (val_rmse={cands[best]['val']['rmse']:.6g})", flush=True)

    finals = [{**cfg, "name": f"final_s{s}", "seed": s} for s in SEEDS]
    recs = run_stage(finals, train_fn, args.workers, RUNS)
    falhas = [n for n, r in recs.items() if "failed" in r]
    if falhas:
        sys.exit(f"seeds com falha: {falhas}; chosen.json não foi gravado")

    rmses = [recs[c["name"]]["val"]["rmse"] for c in finals]
    chosen = {
        "escolha": best,
        "motivo": f"menor val_rmse entre as {N_CANDIDATES} configurações de 4A e 4B; teste não usado",
        "n_configs_tentadas": N_CANDIDATES,
        "val_busca": cands[best]["val"],
        "config": cfg,
        "seeds": list(SEEDS),
        "finais": [
            {"name": c["name"], "seed": c["seed"], "val": recs[c["name"]]["val"],
             "best_epoch": recs[c["name"]]["best_epoch"]}
            for c in finals
        ],
        "val_rmse_media": statistics.mean(rmses),
        "val_rmse_desvio": statistics.stdev(rmses),
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "chosen.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"val_rmse das 3 sementes: média {chosen['val_rmse_media']:.6g}, desvio {chosen['val_rmse_desvio']:.3g}", flush=True)
    print(f"Gravado: {RES / 'chosen.json'}", flush=True)


if __name__ == "__main__":
    main()
