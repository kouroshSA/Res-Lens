#!/usr/bin/env python3
"""
Pooled PaCMAP: all the replicates in ONE projection per layer.

Generated: 2026-09-05 21:00:00

For each layer, the residuals of all ten models are pooled into a single PaCMAP fit
(10 models x 200 pairs = 2000 points per layer), then coloured two ways:

    by CLASS  (PRS vs RRS)   -- is the class structure shared across models?
    by MODEL  (which replicate) -- or does model identity dominate the embedding?

THE CAVEAT THAT DECIDES HOW TO READ THIS
  The ten replicates were trained independently, so their 768-d residual spaces are NOT
  aligned: there is no reason model A's coordinate 5 corresponds to model B's. Pooling
  raw vectors therefore risks an embedding that mostly separates by MODEL, which would
  say nothing about class structure.

  Two things are done about that:
    1. Each model's residuals are z-scored per dimension USING THAT MODEL'S OWN pooled
       mean/sd at that layer, removing per-model location and scale offsets. Rotational
       misalignment between models remains -- that cannot be fixed without a learned
       alignment, and none is assumed here.
    2. Both colourings are always produced, and a quantitative check is reported: the
       silhouette of the embedding with respect to CLASS versus with respect to MODEL.
       Whichever is larger is what the picture is actually showing.

  A pooled embedding that separates by model and not by class is a real (negative)
  answer about cross-model comparability, not a failed plot -- and it is reported as
  such rather than being cropped out.

Usage:
  python pooled_projection.py --residual-root <dir> \
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
from sklearn.metrics import silhouette_score

MODELS = [f"V3-{i}" for i in range(1, 11)]
PRS_COLOR = "darkorange"
RRS_COLOR = "royalblue"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--layers", default="", help="default: all")
    ap.add_argument("--n-neighbors", type=int, default=30)
    args = ap.parse_args()

    from pacmap import PaCMAP

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rroot = Path(args.residual_root)
    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}

    # ---- load every model's residuals -------------------------------------
    P, R = {}, {}
    for m in MODELS:
        d = rroot / f"{m}_iter{picks[m]}"
        if not (d / "prs_residuals.pt").exists():
            print(f"[skip] {m}"); continue
        P[m] = torch.load(d / "prs_residuals.pt").numpy()
        R[m] = torch.load(d / "rrs_residuals.pt").numpy()
    models = list(P.keys())
    n_layers = P[models[0]].shape[1]
    print(f"pooled {len(models)} models, {n_layers} layers, "
          f"{sum(len(P[m]) + len(R[m]) for m in models)} points per layer")

    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers.strip() else list(range(n_layers)))

    cmap = plt.get_cmap("tab10")
    stats = {}
    embeds = {}

    for L in layers:
        blocks, cls_lab, mdl_lab = [], [], []
        for i, m in enumerate(models):
            X = np.vstack((P[m][:, L, :], R[m][:, L, :]))
            # per-model z-score at this layer: removes location/scale offsets between
            # independently trained models. Rotational misalignment is NOT removed.
            mu, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1.0
            blocks.append((X - mu) / sd)
            cls_lab += [1] * len(P[m]) + [0] * len(R[m])
            mdl_lab += [i] * (len(P[m]) + len(R[m]))
        Z = np.vstack(blocks)
        cls_lab = np.array(cls_lab); mdl_lab = np.array(mdl_lab)

        emb = PaCMAP(n_components=2, n_neighbors=args.n_neighbors).fit_transform(Z)
        embeds[L] = (emb, cls_lab, mdl_lab)

        s_cls = float(silhouette_score(emb, cls_lab))
        s_mdl = float(silhouette_score(emb, mdl_lab))
        stats[L] = dict(silhouette_by_class=s_cls, silhouette_by_model=s_mdl,
                        dominant="model" if s_mdl > s_cls else "class")
        print(f"  layer {L:>2}: silhouette(class)={s_cls:+.4f}  "
              f"silhouette(model)={s_mdl:+.4f}  -> dominated by {stats[L]['dominant']}")

    # ---- grid figures ------------------------------------------------------
    for mode in ("class", "model"):
        ncol = 5
        nrow = int(np.ceil(len(layers) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.9 * nrow),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for a in axes[len(layers):]:
            a.axis("off")
        for a, L in zip(axes, layers):
            emb, cl, ml = embeds[L]
            if mode == "class":
                a.scatter(emb[cl == 0, 0], emb[cl == 0, 1], s=3, c=RRS_COLOR,
                          alpha=.45, edgecolors="none", label="RRS")
                a.scatter(emb[cl == 1, 0], emb[cl == 1, 1], s=3, c=PRS_COLOR,
                          alpha=.45, edgecolors="none", label="PRS")
                sub = f"sil(class) = {stats[L]['silhouette_by_class']:+.3f}"
            else:
                for i, m in enumerate(models):
                    a.scatter(emb[ml == i, 0], emb[ml == i, 1], s=3, c=[cmap(i % 10)],
                              alpha=.45, edgecolors="none", label=m)
                sub = f"sil(model) = {stats[L]['silhouette_by_model']:+.3f}"
            a.set_title(f"Layer {L:02d}   {sub}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        h, lbl = axes[0].get_legend_handles_labels()
        fig.legend(h, lbl, loc="lower right", fontsize=8, markerscale=3,
                   ncol=2 if mode == "model" else 1)
        fig.suptitle(f"Pooled PaCMAP of all {len(models)} replicates, "
                     f"coloured by {mode.upper()} — one joint fit per layer "
                     f"(per-model z-scored; rotational alignment NOT assumed)",
                     fontsize=13)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"pooled_pacmap_by_{mode}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out/f'pooled_pacmap_by_{mode}.png'}")

    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    ls = sorted(stats)
    ax.plot(ls, [stats[L]["silhouette_by_class"] for L in ls], "-o", ms=4,
            color="#c1440e", label="silhouette by CLASS (PRS vs RRS)")
    ax.plot(ls, [stats[L]["silhouette_by_model"] for L in ls], "-s", ms=4,
            color="#3b4cc0", label="silhouette by MODEL (which replicate)")
    ax.axhline(0, ls=":", color="0.5")
    ax.set_xlabel("Layer"); ax.set_ylabel("silhouette in the pooled 2-d embedding")
    ax.legend(fontsize=9); ax.grid(alpha=.25)
    ax.set_title("What does the pooled embedding actually separate?", fontsize=11.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pooled_pacmap_what_dominates.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'pooled_pacmap_what_dominates.png'}")

    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()),
               "models": models, "picks": {m: picks[m] for m in models},
               "n_neighbors": args.n_neighbors,
               "preprocessing": "per-model per-dimension z-score at each layer; "
                                "no rotational alignment between models",
               "per_layer": stats},
              open(out / "pooled_pacmap_stats.json", "w"), indent=2)
    print(f"wrote {out/'pooled_pacmap_stats.json'}")


if __name__ == "__main__":
    main()
