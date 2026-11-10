import csv
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time
from functools import lru_cache

from . import ext

DATA = os.path.join(os.path.dirname(__file__), "mm_bench.csv")
MODEL = os.path.join(os.path.dirname(__file__), "mm_model.pkl")
FIELDS = ["chip", "cores", "M1", "N", "M2", "split", "join", "block", "ms"]


@lru_cache(None)
def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    info = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                          capture_output=True, text=True).stdout
    cores = int(re.search(r"Total Number of Cores:\s*(\d+)", info).group(1))
    return chip, cores


def features(M1, N, M2, cores, split, join, block):
    return [math.log2(M1), math.log2(N), math.log2(M2), cores,
            math.log2(split), math.log2(join), math.log2(block),
            math.log2(max(1, M1 // join)), math.log2(ext.cdiv(N, split))]


def time_ms(x1, x2, config, best=None, reps=5):
    import torch
    ext.matrix_mul(x1, x2, config)
    torch.mps.synchronize()
    t = time.perf_counter()
    ext.matrix_mul(x1, x2, config)
    torch.mps.synchronize()
    first = (time.perf_counter() - t) * 1e3
    if best is not None and first > 2 * best:
        return first
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        ext.matrix_mul(x1, x2, config)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t)
    return sorted(times)[reps // 2] * 1e3


def shapes(n=500, seed=0):
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        M1 = round(2 ** rng.uniform(7, 13))
        N = round(2 ** rng.uniform(8, 14))
        M2 = round(2 ** rng.uniform(5, 12))
        if rng.random() < 0.5:
            N = 1 << round(math.log2(N))
        if M1 * N * M2 <= 1 << 31 and M1 * N <= 1 << 25 and N * M2 <= 1 << 25:
            out.add((M1, N, M2))
    return sorted(out, key=lambda s: s[0] * s[1] * s[2])


def bench(n=500):
    import torch
    chip, cores = machine()
    done = {(r["M1"], r["N"], r["M2"]) for r in load()} if os.path.exists(DATA) else set()
    with open(DATA, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(FIELDS)
        for i, (M1, N, M2) in enumerate(shapes(n)):
            if (M1, N, M2) in done:
                continue
            x1 = torch.randn(M1, N, device="mps")
            x2 = torch.randn(N, M2, device="mps")
            best = None
            for split, join, block in ext.mm_configs(M1, N, M2):
                ms = time_ms(x1, x2, (split, join, block), best)
                best = ms if best is None else min(best, ms)
                w.writerow([chip, cores, M1, N, M2, split, join, block, ms])
            f.flush()
            print(f"{i + 1}/{n}  {M1}x{N} @ {N}x{M2}")


def load():
    with open(DATA) as f:
        return [{k: (v if k == "chip" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def row_features(r):
    return features(r["M1"], r["N"], r["M2"], r["cores"], r["split"], r["join"], r["block"])


def train(test_frac=0.2, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    rows = load()
    keys = sorted({(r["M1"], r["N"], r["M2"]) for r in rows})
    random.Random(seed).shuffle(keys)
    test = set(keys[:int(len(keys) * test_frac)])

    def fit(rs):
        X = [row_features(r) for r in rs]
        y = [math.log(r["ms"]) for r in rs]
        return GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05).fit(X, y)

    model = fit([r for r in rows if (r["M1"], r["N"], r["M2"]) not in test])
    by_shape = {}
    for r in rows:
        by_shape.setdefault((r["M1"], r["N"], r["M2"]), []).append(r)
    ratio_model, ratio_default = [], []
    for k in test:
        rs = by_shape[k]
        best = min(r["ms"] for r in rs)
        pred = model.predict([row_features(r) for r in rs])
        chosen = rs[min(range(len(rs)), key=lambda i: pred[i])]
        default = next(r for r in rs if r["split"] == 1 and r["join"] == 1
                       and r["block"] == ext.plan(int(r["N"]))[0])
        ratio_model.append(chosen["ms"] / best)
        ratio_default.append(default["ms"] / best)
    gm = lambda v: math.exp(sum(math.log(a) for a in v) / len(v))
    print(f"held-out shapes: {len(test)}")
    print(f"model   : {gm(ratio_model):.3f}x of best (geo mean), worst {max(ratio_model):.2f}x")
    print(f"default : {gm(ratio_default):.3f}x of best (geo mean), worst {max(ratio_default):.2f}x")

    model = fit(rows)
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)


@lru_cache(None)
def _model():
    if not os.path.exists(MODEL):
        return None
    with open(MODEL, "rb") as f:
        return pickle.load(f)


@lru_cache(4096)
def pick(M1, N, M2):
    model = _model()
    if model is None or 0 in (M1, N, M2):
        return (1, 1, None)
    cands = ext.mm_configs(M1, N, M2)
    cores = machine()[1]
    pred = model.predict([features(M1, N, M2, cores, s, j, b) for s, j, b in cands])
    return cands[min(range(len(cands)), key=lambda i: pred[i])]


if __name__ == "__main__":
    {"bench": bench, "train": train}[sys.argv[1]]()
