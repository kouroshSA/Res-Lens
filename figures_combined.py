#!/usr/bin/env python3
"""
Combined PaCMAP figures: all ten replicates pooled, coloured ONLY by class.

Generated: 2026-09-05 22:00:00

Produces two layouts, each in light and dark:

  A. SINGLE PANEL at the decision layer   -- all ten models in one projection
  B. PER-LAYER GRID (13 panels)           -- one joint fit per layer

Model identity is deliberately not shown. The companion figure coloured by model
(pooled_pacmap_by_model) already documents that model identity dominates layers 0-11
and vanishes at layer 12; these versions are for reading the CLASS structure.

Preprocessing (identical to ppigplm_pooled_pacmap_*): each model's residuals are
z-scored per dimension at each layer using that model's own mean/sd, removing per-model
location and scale. Rotational misalignment between independently trained models is NOT
removed -- no learned alignment is assumed. Silhouette by class is annotated on each
panel so the picture is never read without its number.

Usage:
  python figures_combined.py --residual-root <dir> \
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

STYLES = {
    "light": dict(prs="#E8710A", rrs="#2C5FD6", fg="black", grid_alpha=.25,
                  edge="white", rc={}),
    "dark": dict(prs="#FFA53B", rrs="#6BA8FF", fg="white", grid_alpha=.30,
                 edge="none", rc={"figure.facecolor": "#111417",
                                  "axes.facecolor": "#111417",
                                  "savefig.facecolor": "#111417",
                                  "text.color": "white",
                                  "axes.labelcolor": "white",
                                  "axes.edgecolor": "#8894a0",
                                  "xtick.color": "white",
                                  "ytick.color": "white"}),
}


def build(residual_root, picks):
    rroot = Path(residual_root)
    P, R = {}, {}
    for m in MODELS:
        d = rroot / f"{m}_iter{picks[m]}"
        if not (d / "prs_residuals.pt").exists():
            print(f"[skip] {m}")
            continue
        P[m] = torch.load(d / "prs_residuals.pt").numpy()
        R[m] = torch.load(d / "rrs_residuals.pt").numpy()
    return P, R


def pooled_layer(P, R, L):
    """Per-model z-scored, pooled across models. Returns (Z, class_labels)."""
    blocks, cls = [], []
    for m in P:
        X = np.vstack((P[m][:, L, :], R[m][:, L, :]))
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        blocks.append((X - mu) / sd)
        cls += [1] * len(P[m]) + [0] * len(R[m])
    return np.vstack(blocks), np.array(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--decision-layer", type=int, default=12)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pacmap import PaCMAP

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}
    P, R = build(args.residual_root, picks)
    n_models = len(P)
    n_layers = P[next(iter(P))].shape[1]
    n_pts = sum(len(P[m]) + len(R[m]) for m in P)
    print(f"{n_models} models, {n_layers} layers, {n_pts} points per layer")

    # one embedding per layer, reused by both layouts and both styles
    emb, cls, sil = {}, {}, {}
    for L in range(n_layers):
        Z, y = pooled_layer(P, R, L)
        # random_state fixed: PaCMAP is stochastic, and an unseeded fit changes the
        # 2-d embedding (and therefore its silhouette) between runs. The
        # FULL-DIMENSIONAL metrics in the layer-curve figure are the stable ones;
        # embedding silhouette is projection-dependent and illustrative only.
        e = PaCMAP(n_components=2, n_neighbors=args.n_neighbors,
                   random_state=args.seed).fit_transform(Z)
        emb[L], cls[L] = e, y
        sil[L] = float(silhouette_score(e, y))
        print(f"  layer {L:>2}: silhouette(class) = {sil[L]:+.4f}")

    DL = args.decision_layer

    for style_name, S in STYLES.items():
        with plt.rc_context(S["rc"]):
            # ---- A. single panel at the decision layer --------------------
            fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
            e, y = emb[DL], cls[DL]
            ax.scatter(e[y == 0, 0], e[y == 0, 1], s=13, c=S["rrs"], alpha=.62,
                       edgecolors="none", label="RRS — non-interacting")
            ax.scatter(e[y == 1, 0], e[y == 1, 1], s=13, c=S["prs"], alpha=.62,
                       edgecolors="none", label="PRS — interacting")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"All {n_models} replicates pooled — decision layer "
                         f"(layer {DL})\nsilhouette by class = {sil[DL]:+.3f}",
                         fontsize=13, color=S["fg"])
            ax.legend(fontsize=10, markerscale=1.8, loc="best", framealpha=.85)
            for ext in ("png", "pdf"):
                fig.savefig(out / f"combined_decision_layer_{style_name}.{ext}",
                            dpi=300, bbox_inches="tight")
            plt.close(fig)

            # ---- B. per-layer grid ----------------------------------------
            ncol = 5
            nrow = int(np.ceil(n_layers / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.85 * ncol, 3.75 * nrow),
                                     constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for a in axes[n_layers:]:
                a.axis("off")
            for L in range(n_layers):
                a = axes[L]
                e, y = emb[L], cls[L]
                a.scatter(e[y == 0, 0], e[y == 0, 1], s=3.2, c=S["rrs"], alpha=.5,
                          edgecolors="none", label="RRS — non-interacting")
                a.scatter(e[y == 1, 0], e[y == 1, 1], s=3.2, c=S["prs"], alpha=.5,
                          edgecolors="none", label="PRS — interacting")
                a.set_xticks([]); a.set_yticks([])
                mark = "  ← decision layer" if L == DL else ""
                a.set_title(f"Layer {L:02d}   silhouette {sil[L]:+.3f}{mark}",
                            fontsize=10.5, color=S["fg"])
            h, lbl = axes[0].get_legend_handles_labels()
            fig.legend(h, lbl, loc="lower right", fontsize=11, markerscale=4,
                       framealpha=.85)
            fig.suptitle(f"All {n_models} replicates pooled — PRS vs RRS through the "
                         f"network ({n_pts} pairs per panel)", fontsize=14,
                         color=S["fg"])
            for ext in ("png", "pdf"):
                fig.savefig(out / f"combined_all_layers_{style_name}.{ext}",
                            dpi=220, bbox_inches="tight")
            plt.close(fig)
        print(f"wrote {style_name} versions")

    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()),
               "n_models": n_models, "picks": {m: picks[m] for m in P},
               "n_points_per_layer": n_pts, "decision_layer": DL,
               "n_neighbors": args.n_neighbors, "pacmap_seed": args.seed,
               "preprocessing": "per-model per-dimension z-score at each layer; "
                                "no rotational alignment assumed",
               "silhouette_by_class_per_layer": sil},
              open(out / "combined_pacmap_stats.json", "w"), indent=2)
    print(f"wrote {out/'combined_pacmap_stats.json'}")


if __name__ == "__main__":
    main()
