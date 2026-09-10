#!/usr/bin/env python3
"""
Split-half control for the causal directional ablation — the replicates.

Generated: 2026-09-05 20:00:00

THE GAP THIS CLOSES
  In the ablation reported so far, v = normalize(mean_PRS - mean_RRS) was estimated from
  the SAME 100+100 pairs on which the ablated model was then evaluated. The
  random-direction control makes trivial circularity implausible -- a direction fit to
  nothing does nothing -- but strictly, v having seen the evaluation pairs is not
  excluded as a contributor.

  Here v never sees the pairs it is judged on:

    1. split PRS and RRS each into halves A and B (stratified, seeded)
    2. estimate v_A from half A only  (mean_PRS_A - mean_RRS_A at the decision layer)
    3. ablate the model with v_A
    4. evaluate PRS-vs-RRS AUROC on half B ONLY
    5. repeat with A and B swapped, over many random splits

  Baseline and random-direction controls are evaluated on the SAME held-out half, so
  every comparison is like-for-like.

  If the collapse survives at full strength, the effect is a property of the model, not
  an artifact of fitting the direction to the evaluation data.

EFFICIENCY
  Ablation mutates weights, so each evaluation needs pristine weights. Rather than
  reloading the checkpoint hundreds of times, the original c_proj weights are cloned
  once per model and restored before each ablation.

Usage:
  python splithalf.py --root <picks dir> --residual-root <dir> \
      --picks <checkpoint_picks.csv> --outdir <dir> [--n-splits 10]
"""

import argparse
import csv
import importlib.util
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EXTRACTOR = Path(__file__).with_name("residual_extract.py")
MODELS = [f"V3-{i}" for i in range(1, 11)]


def load_extractor():
    spec = importlib.util.spec_from_file_location("residual_extract", EXTRACTOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def score_subset(model, lines, idx, stoi, tok1, block_size, device):
    out = []
    for j in idx:
        line = lines[j]
        i = line.rfind("<")
        ids = [stoi[c] for c in line[:i + 1]][-block_size:]
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        out.append(F.softmax(logits[0, -1, :].float(), 0)[tok1].item())
    return np.array(out)


def snapshot(model):
    return [(b.attn.c_proj.weight.data.clone(), b.mlp.c_proj.weight.data.clone())
            for b in model.transformer.h]


def restore(model, snap):
    for b, (wa, wm) in zip(model.transformer.h, snap):
        b.attn.c_proj.weight.data.copy_(wa)
        b.mlp.c_proj.weight.data.copy_(wm)


def ablate(model, v, lam=1.0):
    for b in model.transformer.h:
        for mod in (b.attn.c_proj, b.mlp.c_proj):
            W = mod.weight.data
            vv = v.to(W.device, W.dtype)
            mod.weight.data = W - lam * torch.outer(vv, vv @ W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory containing <model>/checkpoints/")
    ap.add_argument("--residual-root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    root, rroot = Path(args.root), Path(args.residual_root)
    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}
    E = load_extractor()
    L = args.layer
    rows = []

    for m in MODELS:
        it = picks[m]
        rd = rroot / f"{m}_iter{it}"
        if not (rd / "prs_residuals.pt").exists():
            print(f"[skip] {m}: residuals missing"); continue
        prs_res = torch.load(rd / "prs_residuals.pt").numpy()
        rrs_res = torch.load(rd / "rrs_residuals.pt").numpy()

        stoi = pickle.load(open(root / m / "meta.pkl", "rb"))["stoi"]
        tok1 = stoi["1"]
        prs_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"PRS-{m}.csv")
        rrs_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"RRS-{m}.csv")
        model, margs, _ = E.load_model(str(root / m / "checkpoints" / f"ckpt_{it}.pt"),
                                       args.device)
        bs, nembd = margs["block_size"], margs["n_embd"]
        snap = snapshot(model)

        nP, nR = len(prs_lines), len(rrs_lines)
        rng = np.random.default_rng(0)
        base_h, abl_h, rnd_h = [], [], []

        for s in range(args.n_splits):
            pp = rng.permutation(nP); rr = rng.permutation(nR)
            halves = [(pp[:nP // 2], rr[:nR // 2]), (pp[nP // 2:], rr[nR // 2:])]
            for fit_i in (0, 1):
                fitP, fitR = halves[fit_i]
                evP, evR = halves[1 - fit_i]

                v = prs_res[fitP, L, :].mean(0) - rrs_res[fitR, L, :].mean(0)
                v = torch.from_numpy(v / np.linalg.norm(v)).float()

                # baseline on the held-out half
                restore(model, snap)
                p = score_subset(model, prs_lines, evP, stoi, tok1, bs, args.device)
                r = score_subset(model, rrs_lines, evR, stoi, tok1, bs, args.device)
                base_h.append(roc_auc_score([1] * len(p) + [0] * len(r),
                                            np.concatenate([p, r])))

                # ablate with v fit on the OTHER half, evaluate held-out
                restore(model, snap)
                ablate(model, v, 1.0)
                p = score_subset(model, prs_lines, evP, stoi, tok1, bs, args.device)
                r = score_subset(model, rrs_lines, evR, stoi, tok1, bs, args.device)
                abl_h.append(roc_auc_score([1] * len(p) + [0] * len(r),
                                           np.concatenate([p, r])))

            if s < args.n_random:      # random control on the same held-out halves
                evP, evR = halves[1]
                g = torch.Generator().manual_seed(1000 + s)
                rv = torch.randn(nembd, generator=g)
                rv = rv / rv.norm()
                restore(model, snap)
                ablate(model, rv, 1.0)
                p = score_subset(model, prs_lines, evP, stoi, tok1, bs, args.device)
                r = score_subset(model, rrs_lines, evR, stoi, tok1, bs, args.device)
                rnd_h.append(roc_auc_score([1] * len(p) + [0] * len(r),
                                           np.concatenate([p, r])))

        restore(model, snap)
        del model
        torch.cuda.empty_cache()

        row = dict(model=m, ckpt=it, n_evals=len(abl_h),
                   base_heldout=float(np.mean(base_h)), base_sd=float(np.std(base_h, ddof=1)),
                   abl_heldout=float(np.mean(abl_h)), abl_sd=float(np.std(abl_h, ddof=1)),
                   rnd_heldout=float(np.mean(rnd_h)), rnd_sd=float(np.std(rnd_h, ddof=1)),
                   gap=float(np.mean(rnd_h) - np.mean(abl_h)))
        rows.append(row)
        print(f"{m:7} ckpt_{it:<5} heldout base={row['base_heldout']:.4f}  "
              f"ablate v(other half)={row['abl_heldout']:.4f}  "
              f"random={row['rnd_heldout']:.4f}  gap={row['gap']:+.4f}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"\nacross {len(rows)} models, HELD-OUT halves only "
          f"({rows[0]['n_evals']} evaluations per model):")
    for k, lab in (("base_heldout", "baseline"), ("abl_heldout", "ablate v (other half)"),
                   ("rnd_heldout", "random direction"), ("gap", "specificity gap")):
        v = col(k)
        print(f"  {lab:26} {v.mean():+.4f} +/- {v.std(ddof=1):.4f}  "
              f"[{v.min():+.4f}, {v.max():+.4f}]")
    print(f"  models below chance after ablation: {(col('abl_heldout') < 0.5).sum()} of {len(rows)}")

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(rows)); w = 0.27
    ax.bar(x - w, col("base_heldout"), w, yerr=col("base_sd"), capsize=2,
           color="#444444", label="baseline (held-out half)")
    ax.bar(x, col("abl_heldout"), w, yerr=col("abl_sd"), capsize=2,
           color="#c1440e", label="ablate $v$ fit on the OTHER half")
    ax.bar(x + w, col("rnd_heldout"), w, yerr=col("rnd_sd"), capsize=2,
           color="#3b4cc0", label="random direction")
    ax.axhline(0.5, ls=":", color="0.4")
    ax.set_xticks(x); ax.set_xticklabels([f"{r['model']}\n{r['ckpt']}" for r in rows],
                                         fontsize=8)
    ax.set_ylabel("PRS-vs-RRS AUROC on held-out half")
    ax.set_ylim(min(0.3, col("abl_heldout").min() - 0.06), 1.0)
    ax.legend(fontsize=9); ax.grid(alpha=.25, axis="y")
    ax.set_title("Split-half control: the class direction never sees the pairs it is "
                 "judged on", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"splithalf_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'splithalf_ablation.png'}")

    with open(out / "splithalf_summary.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()), "layer": L,
               "n_splits": args.n_splits, "picks": picks, "per_model": rows,
               "across_model": {k: dict(mean=float(col(k).mean()),
                                        sd=float(col(k).std(ddof=1)))
                                for k in ("base_heldout", "abl_heldout",
                                          "rnd_heldout", "gap")}},
              open(out / "splithalf_summary.json", "w"), indent=2)
    print(f"wrote {out/'splithalf_summary.csv'} and .json")


if __name__ == "__main__":
    main()
