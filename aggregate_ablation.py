#!/usr/bin/env python3
"""
Cross-model aggregation of the causal directional ablation, the replicates.

Generated: 2026-09-05 19:00:00

Each model is ablated on its OWN eval set at its OWN selected checkpoint (mean-gap
criterion). The operator is W <- W - lambda*v(v^T W) applied to attn.c_proj and
mlp.c_proj in every block, with v the layer-12 class direction.

The number that matters is the SPECIFICITY GAP:

    gap = AUROC(random direction, lambda=1)  -  AUROC(class direction, lambda=1)

A large positive gap means the collapse is specific to the class direction rather than
generic damage from perturbing weights. Reporting the class-direction collapse alone
would prove nothing -- mutilating any model degrades it.

Usage:
  python aggregate_ablation.py \
      --root <tenmodel_ablation_dir> --picks <checkpoint_picks.csv> --outdir <dir>
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

MODELS = [f"V3-{i}" for i in range(1, 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}
    agrees = {r["model"]: r["heldout_agrees"] == "True"
              for r in csv.DictReader(open(args.picks))}

    data = {}
    for m in MODELS:
        f = root / f"{m}_iter{picks[m]}" / "ablation_metrics.json"
        if f.exists():
            data[m] = json.load(open(f))
        else:
            print(f"  [skip] {m}: {f} missing")
    if not data:
        raise SystemExit("no ablation metrics found")
    print(f"loaded {len(data)} models")

    rows = []
    print(f"\n{'model':7} {'ckpt':>6} {'base':>7} {'class l=1':>10} {'random l=1':>18} "
          f"{'gap':>8} {'below chance':>13}")
    for m, d in data.items():
        r = dict(model=m, ckpt=picks[m], heldout_agrees=agrees[m],
                 baseline=d["baseline_auroc_computed"],
                 class_l1=d["auroc_class_lambda1"],
                 random_l1=d["auroc_random_lambda1_mean"],
                 random_sd=d["auroc_random_lambda1_std"],
                 gap=d["specificity_gap"])
        r["below_chance"] = r["class_l1"] < 0.5
        r["drop"] = r["baseline"] - r["class_l1"]
        rows.append(r)
        print(f"{m:7} {picks[m]:>6} {r['baseline']:>7.4f} {r['class_l1']:>10.4f} "
              f"{r['random_l1']:>11.4f}+/-{r['random_sd']:<5.4f} {r['gap']:>+8.4f} "
              f"{str(r['below_chance']):>13}")

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"\nacross {len(rows)} models (mean +/- sd):")
    for k, lab in (("baseline", "baseline AUROC"),
                   ("class_l1", "AUROC after ablating v"),
                   ("random_l1", "AUROC after random direction"),
                   ("gap", "specificity gap"),
                   ("drop", "baseline - ablated")):
        v = col(k)
        print(f"  {lab:30} {v.mean():+.4f} +/- {v.std(ddof=1):.4f}   "
              f"[{v.min():+.4f}, {v.max():+.4f}]")
    nb = int(col("below_chance").sum())
    print(f"  {'models driven BELOW chance':30} {nb} of {len(rows)}")

    # random-direction control must be indistinguishable from baseline
    delta_rand = col("random_l1") - col("baseline")
    print(f"\n  random-direction AUROC minus baseline: "
          f"{delta_rand.mean():+.5f} +/- {delta_rand.std(ddof=1):.5f}  "
          f"(should be ~0 -- confirms the operator alone does no damage)")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [f"{r['model']}\n{r['ckpt']}" for r in rows]
    w = 0.27
    ax[0].bar(x - w, col("baseline"), w, color="#444444", label="baseline")
    ax[0].bar(x, col("class_l1"), w, color="#c1440e", label="ablate class $v$ ($\\lambda$=1)")
    ax[0].bar(x + w, col("random_l1"), w, yerr=col("random_sd"), capsize=2.5,
              color="#3b4cc0", label="random direction ($\\lambda$=1)")
    ax[0].axhline(0.5, ls=":", color="0.4", label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel("PRS-vs-RRS AUROC")
    ax[0].set_ylim(min(0.3, col("class_l1").min() - 0.05), 1.0)
    ax[0].legend(fontsize=8.5, ncol=2)
    ax[0].grid(alpha=.25, axis="y")
    ax[0].set_title("Ablating one direction of 768, per replicate", fontsize=11)

    ax[1].bar(x, col("gap"), color="#1b7837")
    ax[1].axhline(col("gap").mean(), ls="--", color="k",
                  label=f"mean {col('gap').mean():+.3f}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.5)
    ax[1].set_ylabel("specificity gap  (random $-$ class)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=.25, axis="y")
    ax[1].set_title("Specificity of the collapse", fontsize=11)

    fig.suptitle("Causal directional ablation across the replicates, "
                 "each at its selected checkpoint", fontsize=12.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tenmodel_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'tenmodel_ablation.png'}")

    with open(out / "tenmodel_ablation_summary.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()),
               "root": str(root.resolve()), "picks": picks,
               "per_model": rows,
               "across_model": {k: dict(mean=float(col(k).mean()),
                                        sd=float(col(k).std(ddof=1)),
                                        min=float(col(k).min()),
                                        max=float(col(k).max()))
                                for k in ("baseline", "class_l1", "random_l1", "gap", "drop")},
               "n_below_chance": nb,
               "random_minus_baseline_mean": float(delta_rand.mean()),
               "random_minus_baseline_sd": float(delta_rand.std(ddof=1))},
              open(out / "tenmodel_ablation_summary.json", "w"), indent=2)
    print(f"wrote {out/'tenmodel_ablation_summary.csv'} and .json")


if __name__ == "__main__":
    main()
