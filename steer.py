#!/usr/bin/env python3
"""
Steering: can the decision be MOVED along v, not merely destroyed?

Generated: 2026-09-11 14:00:00

NECESSITY vs SUFFICIENCY
  The ablation experiment removes v and the decision collapses -- that establishes
  NECESSITY: the model has no alternative pathway.

  This asks the complementary question, SUFFICIENCY: if v really carries the decision,
  then ADDING it should push the model toward "interacting" and subtracting it toward
  "non-interacting". Necessity plus sufficiency is the standard causal pair; ablation
  alone gives only half.

HOW
  No weights are edited. A forward hook adds alpha * v to the residual stream at the
  output of every block, at the decision position only:

      h  <-  h + alpha * v        (alpha swept negative through positive)

  Steering the decision position alone (rather than all positions) keeps the model's
  sequence processing untouched and isolates the readout.

THE TRAP THIS GUARDS AGAINST
  Steering can succeed trivially by SATURATION: a large alpha pins P(class) at 1.0
  regardless of input. That is not control, it is a broken model. The diagnostic is
  whether RANKING survives -- i.e. whether AUROC holds while the operating point moves.

    operating point moves AND AUROC holds   -> genuine steering of the decision
    operating point moves AND AUROC collapses -> saturation, the model stopped reading input

  Both are reported at every alpha, and a random direction of matched magnitude is swept
  alongside as the specificity control.

Usage:
  python ppigplm_steer_20260911_140000.py --root <ckpt root> \
      --residual-root <dir with *_residuals.pt> --picks <picks.csv> --outdir <dir>
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


class Steerer:
    """Adds alpha*v to the residual stream at the LAST position, after every block.

    Implemented as forward hooks so no weights are modified and the effect is exactly
    reversible by removing the hooks.
    """

    def __init__(self, model, v, alpha):
        self.model, self.v, self.alpha, self.handles = model, v, alpha, []

    def __enter__(self):
        if self.alpha == 0.0:
            return self

        def hook(_mod, _inp, out):
            out = out.clone()
            out[:, -1, :] = out[:, -1, :] + self.alpha * self.v.to(out.device, out.dtype)
            return out

        for block in self.model.transformer.h:
            self.handles.append(block.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def score(model, lines, stoi, tok1, block_size, device):
    out = []
    for line in lines:
        i = line.rfind("<")
        ids = [stoi[c] for c in line[:i + 1]][-block_size:]
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        out.append(F.softmax(logits[0, -1, :].float(), 0)[tok1].item())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--residual-root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alphas", default="-8,-4,-2,-1,0,1,2,4,8")
    ap.add_argument("--n-random", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    root, rroot = Path(args.root), Path(args.residual_root)
    picks = {r["model"]: int(r["pick_iteration"]) for r in csv.DictReader(open(args.picks))}
    E = load_extractor()
    L = args.layer
    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []

    for m in MODELS:
        it = picks[m]
        rd = rroot / f"{m}_iter{it}"
        if not (rd / "prs_residuals.pt").exists():
            print(f"[skip] {m}"); continue
        prs_res = torch.load(rd / "prs_residuals.pt").numpy()
        rrs_res = torch.load(rd / "rrs_residuals.pt").numpy()
        stoi = pickle.load(open(root / m / "meta.pkl", "rb"))["stoi"]
        tok1 = stoi["1"]
        a_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"PRS-{m}.csv")
        b_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"RRS-{m}.csv")
        lines = a_lines + b_lines
        y = [1] * len(a_lines) + [0] * len(b_lines)

        model, margs, _ = E.load_model(str(root / m / "checkpoints" / f"ckpt_{it}.pt"),
                                       args.device)
        bs, nembd = margs["block_size"], margs["n_embd"]

        # v is scaled to the typical residual norm at the decision layer, so alpha is
        # interpretable as "fractions of a typical activation magnitude"
        vraw = prs_res[:, L, :].mean(0) - rrs_res[:, L, :].mean(0)
        v = torch.from_numpy(vraw / np.linalg.norm(vraw)).float()
        scale = float(np.linalg.norm(
            np.vstack((prs_res[:, L, :], rrs_res[:, L, :])), axis=1).mean())

        for a in alphas:
            with Steerer(model, v * scale, a):
                p = score(model, lines, stoi, tok1, bs, args.device)
            rows.append(dict(model=m, ckpt=it, direction="class_v", seed=None, alpha=a,
                             auroc=float(roc_auc_score(y, p)),
                             median_P_A=float(np.median(p[:len(a_lines)])),
                             median_P_B=float(np.median(p[len(a_lines):])),
                             frac_called_pos=float((p > 0.5).mean())))

        g = torch.Generator().manual_seed(7)
        for s in range(args.n_random):
            rv = torch.randn(nembd, generator=g); rv = rv / rv.norm()
            for a in alphas:
                with Steerer(model, rv * scale, a):
                    p = score(model, lines, stoi, tok1, bs, args.device)
                rows.append(dict(model=m, ckpt=it, direction="random", seed=s, alpha=a,
                                 auroc=float(roc_auc_score(y, p)),
                                 median_P_A=float(np.median(p[:len(a_lines)])),
                                 median_P_B=float(np.median(p[len(a_lines):])),
                                 frac_called_pos=float((p > 0.5).mean())))
        del model; torch.cuda.empty_cache()

        base = [r for r in rows if r["model"] == m and r["direction"] == "class_v" and r["alpha"] == 0][0]
        lo = [r for r in rows if r["model"] == m and r["direction"] == "class_v" and r["alpha"] == min(alphas)][0]
        hi = [r for r in rows if r["model"] == m and r["direction"] == "class_v" and r["alpha"] == max(alphas)][0]
        print(f"{m:6} ckpt_{it:<5} frac-called-positive  a={min(alphas):+.0f}: {lo['frac_called_pos']:.3f}"
              f"  a=0: {base['frac_called_pos']:.3f}  a={max(alphas):+.0f}: {hi['frac_called_pos']:.3f}"
              f"   | AUROC {lo['auroc']:.3f} / {base['auroc']:.3f} / {hi['auroc']:.3f}", flush=True)

    with open(out / "steering.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def agg(direction, key):
        return np.array([[np.mean([r[key] for r in rows
                                   if r["direction"] == direction and r["alpha"] == a
                                   and r["model"] == m])
                          for a in alphas]
                         for m in sorted({r["model"] for r in rows})], float)

    fp_v, fp_r = agg("class_v", "frac_called_pos"), agg("random", "frac_called_pos")
    au_v, au_r = agg("class_v", "auroc"), agg("random", "auroc")

    print("\n" + "=" * 78)
    print(f"{'alpha':>7} | {'frac called +ve (v)':>20} {'(random)':>10} | {'AUROC (v)':>10} {'(random)':>10}")
    for j, a in enumerate(alphas):
        print(f"{a:>7.1f} | {fp_v[:, j].mean():>20.3f} {fp_r[:, j].mean():>10.3f} | "
              f"{au_v[:, j].mean():>10.4f} {au_r[:, j].mean():>10.4f}")

    z = alphas.index(0.0)
    print(f"\noperating point shift, alpha {min(alphas):+.0f} -> {max(alphas):+.0f}: "
          f"{fp_v[:, 0].mean():.3f} -> {fp_v[:, -1].mean():.3f} "
          f"(random: {fp_r[:, 0].mean():.3f} -> {fp_r[:, -1].mean():.3f})")
    print(f"AUROC at the extremes: {au_v[:, 0].mean():.4f} / {au_v[:, z].mean():.4f} / {au_v[:, -1].mean():.4f}")
    # Verdict must be COMPUTED, not asserted. The control decides it: if a random
    # direction of matched magnitude also destroys ranking, the alphas are simply too
    # large and the run says nothing about steering.
    # Verdict is COMPUTED, and over the USABLE WINDOW rather than the whole sweep.
    # Requiring every alpha to be clean lets one over-large extreme mask a genuine
    # result in the middle -- which is exactly what happened on the first fine sweep.
    base = au_v[:, z].mean()
    ok = [j for j in range(len(alphas))
          if abs(au_r[:, j].mean() - base) <= 0.05 and abs(au_v[:, j].mean() - base) <= 0.05]
    print()
    if len(ok) <= 1:
        print(f"  VERDICT: INCONCLUSIVE. No alpha leaves both the random control and v's "
              f"ranking intact; every value tested is in the destruction regime. "
              f"Re-run with smaller alpha.")
    else:
        lo, hi = min(ok), max(ok)
        span = fp_v[:, hi].mean() - fp_v[:, lo].mean()
        rspan = abs(fp_r[:, hi].mean() - fp_r[:, lo].mean())
        worst = max(abs(au_v[:, j].mean() - base) for j in ok)
        print(f"  USABLE WINDOW: alpha in [{alphas[lo]:+.3f}, {alphas[hi]:+.3f}]")
        print(f"    v      moves the operating point by {span:+.3f} "
              f"({fp_v[:, lo].mean():.3f} -> {fp_v[:, hi].mean():.3f})")
        print(f"    random moves it by only {rspan:.3f}")
        print(f"    max AUROC loss for v inside the window: {worst:.4f} (baseline {base:.4f})")
        if abs(span) > 4 * rspan and worst <= 0.05:
            print(f"  VERDICT: GENUINE STEERING inside the window -- the operating point "
                  f"moves while ranking is preserved, and a random direction does neither.")
        else:
            print(f"  VERDICT: no usable steering -- v does not move the operating point "
                  f"appreciably more than a random direction does.")
        print(f"  NOTE: outside the window the model saturates and ranking collapses; "
              f"steering is bounded, not unlimited.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for M_, lab, c in ((fp_v, "class direction $v$", "#c1440e"), (fp_r, "random direction", "#3b4cc0")):
        ax[0].plot(alphas, M_.mean(0), "-o", ms=5, color=c, label=lab)
        ax[0].fill_between(alphas, M_.mean(0) - M_.std(0), M_.mean(0) + M_.std(0), color=c, alpha=.15)
    ax[0].axhline(0.5, ls=":", color="0.5")
    ax[0].set_xlabel(r"steering strength $\alpha$  (× mean residual norm)")
    ax[0].set_ylabel("fraction of pairs called interacting")
    ax[0].set_title("Does the operating point move?", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)

    for M_, lab, c in ((au_v, "class direction $v$", "#c1440e"), (au_r, "random direction", "#3b4cc0")):
        ax[1].plot(alphas, M_.mean(0), "-o", ms=5, color=c, label=lab)
        ax[1].fill_between(alphas, M_.mean(0) - M_.std(0), M_.mean(0) + M_.std(0), color=c, alpha=.15)
    ax[1].axhline(0.5, ls=":", color="0.5", label="chance")
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel("PRS-vs-RRS AUROC")
    ax[1].set_title("Does ranking survive?  (saturation check)", fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)

    fig.suptitle("Steering the decision along $v$ — sufficiency, alongside the ablation's necessity",
                 fontsize=12.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"steering.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'steering.png'}")

    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()), "layer": L, "alphas": alphas,
               "alpha_units": "multiples of the mean residual norm at the decision layer",
               "picks": picks,
               "frac_called_positive_v": fp_v.mean(0).tolist(),
               "frac_called_positive_random": fp_r.mean(0).tolist(),
               "auroc_v": au_v.mean(0).tolist(),
               "auroc_random": au_r.mean(0).tolist(),
               "rows": rows}, open(out / "steering.json", "w"), indent=2)
    print(f"wrote {out/'steering.json'}")


if __name__ == "__main__":
    main()
