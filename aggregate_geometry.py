#!/usr/bin/env python3
"""
Cross-model aggregation of the the replicates residual-stream analysis.

Generated: 2026-09-05 18:00:00

Aggregates the per-model runs of residual_extract.py across the
replicates, each at its OWN selected checkpoint (mean-gap criterion,
<selection run>/checkpoint_picks.csv -- NOT iter6000, which is
optimal for one replicate only).

Produces:
  1. Layer-resolved curves with one line per model plus the across-model mean, for
     silhouette, |cos(v, PC1)| and variance fraction along v. The question is whether
     the "class direction rotates into the principal axis in the deep layers" result
     is a property of the architecture/task or a quirk of one replicate.
  2. A 2x5 grid of layer-12 PaCMAP projections, one panel per model.
  3. A per-model table at the decision layer.

Each model is evaluated on its OWN held-out eval set (PRS-RRS, 100/100), since the ten
replicates have different splits. Models whose held-out checkpoint choice disagreed
with the training-positive argmax (where flagged) are flagged.

Usage:
  python aggregate_geometry.py --root <tenmodel_residuals_dir> \
      --picks <checkpoint_picks.csv> --outdir <dir>
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
import torch

MODELS = [f"V3-{i}" for i in range(1, 11)]
PRS_COLOR = "darkorange"
RRS_COLOR = "royalblue"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pacmap-layer", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}
    agrees = {r["model"]: r["heldout_agrees"] == "True"
              for r in csv.DictReader(open(args.picks))}

    data = {}
    for m in MODELS:
        d = root / f"{m}_iter{picks[m]}" / "metrics.json"
        if not d.exists():
            print(f"  [skip] {m}: {d} not found")
            continue
        data[m] = json.load(open(d))
    print(f"loaded {len(data)} models")
    if not data:
        raise SystemExit("no per-model metrics found")

    # Derive the layer count from the arrays themselves rather than a metadata key,
    # so this stays correct regardless of which extractor version wrote the file.
    lens = {len(d["silhouette_by_layer"]) for d in data.values()}
    if len(lens) != 1:
        raise SystemExit(f"models disagree on layer count: {lens}")
    n_layers = lens.pop()
    layers = list(range(n_layers))
    L = n_layers - 1

    # ---- per-model table ---------------------------------------------------
    rows = []
    print(f"\n{'model':7} {'ckpt':>6} {'agree':>6} {'AUROC':>7} {'sil@L':>7} "
          f"{'cos(m)':>7} {'varfrac':>8} {'|cos v,PC1|':>12} {'peak layer':>10}")
    for m, d in data.items():
        r = dict(model=m, ckpt=picks[m], heldout_agrees=agrees[m],
                 auroc=d["auroc"],
                 silhouette_L=d["silhouette_by_layer"][L],
                 cos_means_L=d["class_mean_cosine_by_layer"][L],
                 var_frac_L=d["variance_fraction_along_v_by_layer"][L],
                 cos_pc1_L=d["abs_cos_v_pc1_by_layer"][L],
                 peak_layer=int(np.argmax(d["silhouette_by_layer"])),
                 prs_median_P1=d["prs_median_P1"], rrs_median_P1=d["rrs_median_P1"])
        rows.append(r)
        print(f"{m:7} {picks[m]:>6} {str(agrees[m]):>6} {r['auroc']:>7.4f} "
              f"{r['silhouette_L']:>7.4f} {r['cos_means_L']:>7.4f} {r['var_frac_L']:>8.4f} "
              f"{r['cos_pc1_L']:>12.4f} {r['peak_layer']:>10}")

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"\nacross {len(rows)} models at the decision layer (mean +/- sd):")
    for k, lab in (("auroc", "AUROC"), ("silhouette_L", "silhouette"),
                   ("var_frac_L", "variance fraction along v"),
                   ("cos_pc1_L", "|cos(v, PC1)|")):
        v = col(k)
        print(f"  {lab:28} {v.mean():.4f} +/- {v.std(ddof=1):.4f}   "
              f"[{v.min():.4f}, {v.max():.4f}]")
    pk = col("peak_layer")
    print(f"  {'peak silhouette layer':28} "
          f"{int(pk.min())}-{int(pk.max())}, mode {int(np.bincount(pk.astype(int)).argmax())}")

    # ---- layer curves ------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    panels = [("silhouette_by_layer", "Silhouette score", "PRS vs RRS separation"),
              ("abs_cos_v_pc1_by_layer", "|cos(v, PC1)|", "Class direction vs principal axis"),
              ("variance_fraction_along_v_by_layer", "variance fraction along v",
               "Share of variance on the class axis")]
    for a, (key, ylab, title) in zip(ax, panels):
        M = []
        for i, (m, d) in enumerate(data.items()):
            y = d[key]
            M.append(y)
            a.plot(layers, y, "-", lw=1.0, alpha=.45, color=cmap(i % 10), label=m)
        M = np.array(M)
        a.plot(layers, M.mean(0), "-o", lw=2.4, ms=4.5, color="k", label="mean", zorder=5)
        a.fill_between(layers, M.mean(0) - M.std(0), M.mean(0) + M.std(0),
                       color="k", alpha=.12, zorder=1)
        a.set_xlabel("Layer"); a.set_ylabel(ylab); a.set_title(title, fontsize=11)
        a.grid(alpha=.25)
    ax[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("the replicates — residual-stream geometry across ten replicates, "
                 "each at its selected checkpoint", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tenmodel_layer_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'tenmodel_layer_curves.png'}")

    # ---- PaCMAP grid at the decision layer ---------------------------------
    try:
        from pacmap import PaCMAP
    except ImportError:
        print("pacmap unavailable -- skipping grid")
        PaCMAP = None

    if PaCMAP is not None:
        PL = args.pacmap_layer
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        for i, m in enumerate([r["model"] for r in rows]):
            a = axes.flat[i]
            md = root / f"{m}_iter{picks[m]}"
            p = torch.load(md / "prs_residuals.pt").numpy()[:, PL, :]
            q = torch.load(md / "rrs_residuals.pt").numpy()[:, PL, :]
            X = np.vstack((p, q))
            nn = max(5, min(30, len(X) // 6))
            emb = PaCMAP(n_components=2, n_neighbors=nn).fit_transform(X)
            a.scatter(emb[len(p):, 0], emb[len(p):, 1], s=10, c=RRS_COLOR, alpha=.6,
                      edgecolors="none", label="RRS")
            a.scatter(emb[:len(p), 0], emb[:len(p), 1], s=10, c=PRS_COLOR, alpha=.6,
                      edgecolors="none", label="PRS")
            sil = [r for r in rows if r["model"] == m][0]["silhouette_L"]
            au = [r for r in rows if r["model"] == m][0]["auroc"]
            flag = "" if agrees[m] else "  (ckpt tie-break)"
            a.set_title(f"{m}  ckpt_{picks[m]}{flag}\nAUROC {au:.3f}   silhouette {sil:.3f}",
                        fontsize=9.5)
            a.set_xticks([]); a.set_yticks([])
        axes.flat[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"Layer {PL} residual vectors, PaCMAP — the replicates "
                     f"(each fit independently; axes are not comparable across panels)",
                     fontsize=13)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"tenmodel_pacmap_layer{PL:02d}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out/f'tenmodel_pacmap_layer{PL:02d}.png'}")

    with open(out / "tenmodel_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()),
               "root": str(root.resolve()), "picks": picks, "heldout_agrees": agrees,
               "decision_layer": L, "per_model": rows,
               "across_model": {k: dict(mean=float(col(k).mean()),
                                        sd=float(col(k).std(ddof=1)),
                                        min=float(col(k).min()),
                                        max=float(col(k).max()))
                                for k in ("auroc", "silhouette_L", "var_frac_L",
                                          "cos_pc1_L")}},
              open(out / "tenmodel_summary.json", "w"), indent=2)
    print(f"wrote {out/'tenmodel_summary.csv'} and .json")


if __name__ == "__main__":
    main()
