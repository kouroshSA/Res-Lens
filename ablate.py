#!/usr/bin/env python3
"""
Causal directional ablation for nanoGPT-style models.

Generated: 2026-09-05 14:00:00

QUESTION
  The residual analysis showed that at the decision layer the class-mean-difference
  direction v = normalize(mean_PRS - mean_RRS) IS the principal axis
  (see residual_extract.py). That is CORRELATIONAL: PCA finds an axis
  along which the classes happen to separate. This script asks whether the model's
  discrimination actually DEPENDS on it.

OPERATOR (the abliteration operator, Arditi et al. 2024)
  For every module that writes into the residual stream:

      W  <-  W - lambda * v (v^T W)

  With lambda = 1 this is (I - v v^T) W: the module can no longer emit ANY component
  along v. nanoGPT's residual writers are Block.attn.c_proj and Block.mlp.c_proj
  (nanoGPT: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))). nn.Linear stores weight as
  (out_features, in_features) and computes y = x @ W.T, so out_features is the
  residual dimension and left-multiplying by (I - v v^T) removes v from the output.

CONTROLS -- the reason this is an experiment and not a demonstration
  A drop in AUROC after ablation proves nothing on its own: mutilating any model
  degrades it. This script therefore also ablates:
    * RANDOM unit directions (several seeds) at the same lambda
    * PC1 of the RRS class alone (a high-variance direction not defined by the
      class contrast)
  If v collapses AUROC while random directions of equal magnitude do not, the effect
  is specific to the class direction. If random directions collapse it too, the
  result is damage, not mechanism, and must be reported as such.

  A lambda sweep (0 -> 1) additionally shows dose-response, which a single point
  cannot.

Readout is imported from the extractor module so the encoding/truncation/position
convention cannot drift between the two scripts.

Usage:
  python ablate.py \
      --ckpt .../<model>/checkpoints/ckpt_6000.pt --meta .../<model>/meta.pkl \
      --prs .../PRS-<model>.csv --rrs .../RRS-<model>.csv \
      --residual-dir <dir with prs_residuals.pt/rrs_residuals.pt> \
      --outdir <dir> [--baseline-auroc <recorded value>]
"""

import argparse
import copy
import importlib.util
import json
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


def load_extractor():
    if not EXTRACTOR.exists():
        raise SystemExit(f"extractor module not found: {EXTRACTOR}")
    spec = importlib.util.spec_from_file_location("residual_extract", EXTRACTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def score(model, lines, stoi, tok1, block_size, device):
    """P(token '1') at the decision position. Same convention as the extractor."""
    out = []
    for line in lines:
        i = line.rfind("<")
        ids = [stoi[c] for c in line[:i + 1]][-block_size:]
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        out.append(F.softmax(logits[0, -1, :].float(), 0)[tok1].item())
    return np.array(out)


def ablate_(model, v_per_layer, lam):
    """In-place W <- W - lam * v (v^T W) on both residual writers of every block.

    v_per_layer[i] is the unit direction to remove at block i.
    """
    for i, block in enumerate(model.transformer.h):
        v = v_per_layer[i]
        if v is None:
            continue
        for mod in (block.attn.c_proj, block.mlp.c_proj):
            W = mod.weight.data                      # (out=n_embd, in)
            vv = v.to(W.device, W.dtype)
            mod.weight.data = W - lam * torch.outer(vv, vv @ W)


def evaluate(model_factory, v_per_layer, lam, prs_lines, rrs_lines,
             stoi, tok1, block_size, device):
    m = model_factory()
    if lam > 0:
        ablate_(m, v_per_layer, lam)
    p = score(m, prs_lines, stoi, tok1, block_size, device)
    r = score(m, rrs_lines, stoi, tok1, block_size, device)
    au = roc_auc_score([1] * len(p) + [0] * len(r), np.concatenate([p, r]))
    del m
    torch.cuda.empty_cache()
    return float(au), float(np.median(p)), float(np.median(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--prs", required=True)
    ap.add_argument("--rrs", required=True)
    ap.add_argument("--residual-dir", required=True,
                    help="Directory containing prs_residuals.pt / rrs_residuals.pt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baseline-auroc", type=float, default=None)
    ap.add_argument("--direction-layer", type=int, default=12,
                    help="Layer whose class direction is used everywhere (default 12). "
                         "-1 = use each block's own direction.")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--n-random", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    E = load_extractor()
    device = args.device

    import pickle
    stoi = pickle.load(open(args.meta, "rb"))["stoi"]
    tok1 = stoi["1"]
    prs_lines, rrs_lines = E.read_lines(args.prs), E.read_lines(args.rrs)

    ref, margs, iter_num = E.load_model(args.ckpt, device)
    n_layer, n_embd, block_size = margs["n_layer"], margs["n_embd"], margs["block_size"]
    print(f"model iter={iter_num} n_layer={n_layer} n_embd={n_embd}")
    del ref
    torch.cuda.empty_cache()

    def factory():
        m, _, _ = E.load_model(args.ckpt, device)
        return m

    prs_res = torch.load(Path(args.residual_dir) / "prs_residuals.pt")
    rrs_res = torch.load(Path(args.residual_dir) / "rrs_residuals.pt")

    def unit(t):
        return t / t.norm()

    # Class direction. Residual index L corresponds to the output of block L-1,
    # so block i should have the direction measured at residual index i+1.
    if args.direction_layer == -1:
        v_class = [unit(prs_res[:, i + 1, :].mean(0) - rrs_res[:, i + 1, :].mean(0))
                   for i in range(n_layer)]
        dir_tag = "per-layer"
    else:
        L = args.direction_layer
        v = unit(prs_res[:, L, :].mean(0) - rrs_res[:, L, :].mean(0))
        v_class = [v] * n_layer
        dir_tag = f"layer{L}"
    print(f"class direction: {dir_tag}")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = []

    print("\n=== CLASS DIRECTION v ===")
    for lam in lambdas:
        au, mp, mr = evaluate(factory, v_class, lam, prs_lines, rrs_lines,
                              stoi, tok1, block_size, device)
        rows.append(dict(direction="class_v", seed=None, lam=lam,
                         auroc=au, prs_median_P=mp, rrs_median_P=mr))
        print(f"  lambda={lam:4.2f}  AUROC={au:.4f}  medP(PRS)={mp:.4f}  medP(RRS)={mr:.4f}")

    print("\n=== RANDOM DIRECTION CONTROLS (lambda=1.0) ===")
    g = torch.Generator().manual_seed(0)
    for s in range(args.n_random):
        rv = torch.randn(n_embd, generator=g)
        rv = unit(rv)
        au, mp, mr = evaluate(factory, [rv] * n_layer, 1.0, prs_lines, rrs_lines,
                              stoi, tok1, block_size, device)
        rows.append(dict(direction="random", seed=s, lam=1.0,
                         auroc=au, prs_median_P=mp, rrs_median_P=mr))
        print(f"  seed={s}  AUROC={au:.4f}  medP(PRS)={mp:.4f}  medP(RRS)={mr:.4f}")

    print("\n=== RRS-ONLY PC1 CONTROL (high-variance, not class-defined; lambda=1.0) ===")
    X = rrs_res[:, args.direction_layer if args.direction_layer != -1 else n_layer, :].numpy()
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = unit(torch.from_numpy(vt[0].copy()).float())
    au, mp, mr = evaluate(factory, [pc1] * n_layer, 1.0, prs_lines, rrs_lines,
                          stoi, tok1, block_size, device)
    rows.append(dict(direction="rrs_pc1", seed=None, lam=1.0,
                     auroc=au, prs_median_P=mp, rrs_median_P=mr))
    cos_vc = float(abs(torch.dot(pc1, v_class[-1])))
    print(f"  AUROC={au:.4f}  medP(PRS)={mp:.4f}  medP(RRS)={mr:.4f}  |cos(pc1_rrs, v)|={cos_vc:.3f}")

    # ---- summary ---------------------------------------------------------
    base = [r for r in rows if r["direction"] == "class_v" and r["lam"] == 0.0]
    base_au = base[0]["auroc"] if base else None
    full = [r for r in rows if r["direction"] == "class_v" and r["lam"] == 1.0][0]["auroc"]
    rnd = [r["auroc"] for r in rows if r["direction"] == "random"]
    print("\n" + "=" * 70)
    if base_au is not None:
        print(f"baseline (lambda=0)            AUROC = {base_au:.4f}")
        if args.baseline_auroc is not None:
            d = abs(base_au - args.baseline_auroc)
            print(f"  vs recorded {args.baseline_auroc:.4f} -> |delta|={d:.4f} "
                  f"{'PASS' if d <= 0.05 else 'FAIL'}")
    print(f"ablate class direction (l=1)   AUROC = {full:.4f}")
    print(f"ablate random directions (l=1) AUROC = {np.mean(rnd):.4f} +/- {np.std(rnd):.4f}  (n={len(rnd)})")
    specific = (np.mean(rnd) - full)
    print(f"\nspecificity gap (random - class) = {specific:+.4f}")
    print("  A large positive gap means the collapse is SPECIFIC to the class direction.")
    print("  A gap near zero means the effect is generic damage, not mechanism.")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    cv = [r for r in rows if r["direction"] == "class_v"]
    # Ablation can drive AUROC BELOW chance (signal inverted, not merely removed),
    # so the axis floor must follow the data or those points vanish off-plot.
    lo = min([r["auroc"] for r in rows]) - 0.06
    ylo = min(0.45, lo)
    ax[0].plot([r["lam"] for r in cv], [r["auroc"] for r in cv], "-o", ms=5,
               color="#c1440e", label="ablate class direction v")
    ax[0].axhline(np.mean(rnd), ls="--", color="#3b4cc0",
                  label=f"random direction (l=1), mean of {len(rnd)}")
    ax[0].fill_between([0, 1], np.mean(rnd) - np.std(rnd), np.mean(rnd) + np.std(rnd),
                       color="#3b4cc0", alpha=.15)
    ax[0].axhline(0.5, ls=":", color="0.5", label="chance")
    ax[0].set_xlabel("ablation strength $\\lambda$"); ax[0].set_ylabel("PRS-vs-RRS AUROC")
    ax[0].set_ylim(ylo, 1.0); ax[0].grid(alpha=.25); ax[0].legend(fontsize=8.5)
    ax[0].set_title("Dose-response", fontsize=11)

    labels = ["baseline\n$\\lambda$=0", "class v\n$\\lambda$=1", f"random\n$\\lambda$=1",
              "RRS PC1\n$\\lambda$=1"]
    vals = [base_au, full, float(np.mean(rnd)),
            [r["auroc"] for r in rows if r["direction"] == "rrs_pc1"][0]]
    errs = [0, 0, float(np.std(rnd)), 0]
    ax[1].bar(labels, vals, yerr=errs, capsize=4,
              color=["#444444", "#c1440e", "#3b4cc0", "#1b7837"])
    for xi, vv in enumerate(vals):
        ax[1].text(xi, vv + 0.012, f"{vv:.3f}", ha="center", fontsize=9)
    ax[1].axhline(0.5, ls=":", color="0.5")
    ax[1].set_ylabel("PRS-vs-RRS AUROC"); ax[1].set_ylim(ylo, 1.0)
    ax[1].grid(alpha=.25, axis="y")
    ax[1].set_title("Ablation vs controls", fontsize=11)

    fig.suptitle(f"Causal directional ablation — iter{iter_num} "
                 f"({dir_tag} direction, attn.c_proj + mlp.c_proj, all {n_layer} blocks)",
                 fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ppigplm_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'ppigplm_ablation.png'}")

    json.dump({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()), "iter_num": iter_num,
        "direction": dir_tag, "n_prs": len(prs_lines), "n_rrs": len(rrs_lines),
        "baseline_auroc_recorded": args.baseline_auroc,
        "baseline_auroc_computed": base_au,
        "auroc_class_lambda1": full,
        "auroc_random_lambda1_mean": float(np.mean(rnd)),
        "auroc_random_lambda1_std": float(np.std(rnd)),
        "specificity_gap": float(specific),
        "rows": rows,
    }, open(out / "ablation_metrics.json", "w"), indent=2)
    print(f"wrote {out/'ablation_metrics.json'}")


if __name__ == "__main__":
    main()
