#!/usr/bin/env python3
"""
Error-margin analysis: is model error detectable from
residual geometry, WITHOUT the ground-truth label?

Generated: 2026-09-05 15:00:00

FRAMING (and the circularity it avoids)
  Defining "error" as P<0.5 for PRS makes P a perfect predictor of error by
  construction -- a useless baseline. The deployment-relevant question conditions on
  what the model actually SAID:

    Task A  among pairs called INTERACTING (P > 0.5):     true positive vs FALSE POSITIVE
    Task B  among pairs called NON-INTERACTING (P < 0.5): true negative vs FALSE NEGATIVE

  In both, every sample carries the same verdict from the model, so P is no longer
  determinative and the task is genuinely non-trivial.

THE DECOMPOSITION
  At layer 12 the class direction v is the principal axis (|cos(v,PC1)|=0.994) and
  ablating it collapses AUROC 0.912 -> 0.390, so the projection r.v essentially IS
  the model's decision variable. Each residual is split into

      r  =  (r . v) v   +   r_perp

  and the question becomes whether anything BEYOND the decision variable r.v carries
  information about correctness.

  If it does, the model is leaving usable signal on the table and errors are flaggable
  without labels. If it does not, error is not linearly detectable and the honest
  answer is null.

THE CONTROL (mirrors the random-direction control that validated the ablation)
  Three feature sets, identical pipeline:
    baseline    [r . v]              the decision variable alone
    full        PCA of the full r    does the whole residual beat the projection?
    orth_indep  PCA of r_perp, with v estimated from DISJOINT samples
  Only if one BEATS the baseline is there information the decision axis does not
  already carry. A permutation null (labels shuffled) gives the p-value for each.

  A NOTE ON A TRAP, since a first version of this script fell into it:
  a naive "orthogonal" mode that removes the fold-local v is guaranteed to return
  AUROC 0.5 here. v is defined as the class-mean difference, and the label in these
  tasks IS the class, so projecting v out zeroes the training class-mean difference
  exactly, the logistic solver terminates at n_iter=0, and the result is an artifact
  of construction rather than evidence of absence. "orth_indep" estimates v from
  samples outside the task, which breaks the exact cancellation.

PRE-SPECIFIED PIPELINE -- NO TUNING
  StandardScaler -> PCA(n=10) -> LogisticRegression(C=1, balanced).
  Hyperparameters are fixed in advance, so the reported AUROC cannot be inflated by
  selection. Scaler and PCA are fit INSIDE each training fold (leakage control).
  v is recomputed per outer fold from training samples only, so the direction never
  sees held-out data.

  With ~15-18 errors in 768 dimensions, overfitting is the dominant risk. Everything
  above is aimed at that. Read the permutation p-value, not the raw AUROC.

Usage:
  python error_margin.py \
      --ckpt ... --meta ... --prs ... --rrs ... \
      --residual-dir <dir with *_residuals.pt> --outdir <dir> [--layer 12]
"""

import argparse
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXTRACTOR = Path(__file__).with_name("residual_extract.py")
N_PCA = 10
C_REG = 1.0
N_REPEATS = 10
N_PERM = 500


def load_extractor():
    spec = importlib.util.spec_from_file_location("residual_extract", EXTRACTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def score(model, lines, stoi, tok1, block_size, device):
    out = []
    for line in lines:
        i = line.rfind("<")
        ids = [stoi[c] for c in line[:i + 1]][-block_size:]
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        out.append(F.softmax(logits[0, -1, :].float(), 0)[tok1].item())
    return np.array(out)


def make_pipeline():
    return Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=N_PCA, random_state=0)),
        ("lr", LogisticRegression(C=C_REG, max_iter=2000, class_weight="balanced")),
    ])


def cv_auroc(X_res, y, cls_idx, prs_res, rrs_res, layer, feature, seed,
             v_indep=None):
    """One repeat of stratified 5-fold CV, returning pooled out-of-fold AUROC.

    cls_idx[i] = (0 for PRS, 1 for RRS, original row index) for sample i.

    feature:
      "baseline"    [r . v]      the decision variable alone (1 feature)
      "full"        PCA of the FULL residual r
      "orth_indep"  PCA of r_perp, where v is estimated from DISJOINT samples

    WHY THERE IS NO PLAIN "orthogonal" MODE
      Removing the fold-local v = normalize(mean_PRS_train - mean_RRS_train) makes the
      training class-mean difference in r_perp EXACTLY zero:
          mean(r_perp|PRS) - mean(r_perp|RRS) = (mp-mr) - ||mp-mr|| v = 0
      In these tasks the label IS the PRS/RRS class, so that deletes precisely the
      signal being asked about. A mean-difference classifier then has zero gradient at
      the origin, terminates at n_iter=0, and returns AUROC 0.5 by construction --
      an artifact, not evidence of absence. "orth_indep" avoids this by estimating v
      from samples that are not in this task at all, so no exact cancellation occurs.
    """
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.nan
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X_res, y):
        tr_prs = [cls_idx[i][1] for i in tr if cls_idx[i][0] == 0]
        tr_rrs = [cls_idx[i][1] for i in tr if cls_idx[i][0] == 1]
        if len(tr_prs) < 2 or len(tr_rrs) < 2:
            return np.nan
        mp = prs_res[tr_prs, layer, :].mean(0)
        mr = rrs_res[tr_rrs, layer, :].mean(0)
        v = (mp - mr)
        v = v / np.linalg.norm(v)

        if feature == "baseline":
            F_all = (X_res @ v).reshape(-1, 1)
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=C_REG, max_iter=2000,
                                                       class_weight="balanced"))])
        elif feature == "full":
            F_all = X_res
            pipe = make_pipeline()
        elif feature == "orth_indep":
            if v_indep is None:
                return np.nan
            F_all = X_res - np.outer(X_res @ v_indep, v_indep)
            pipe = make_pipeline()
        else:
            raise ValueError(feature)
        pipe.fit(F_all[tr], y[tr])
        oof[te] = pipe.predict_proba(F_all[te])[:, 1]
    return roc_auc_score(y, oof)


def run_task(name, X_res, y, cls_idx, prs_res, rrs_res, layer, rng, v_indep=None):
    res = {}
    for feature in ("baseline", "full", "orth_indep"):
        obs = np.nanmean([cv_auroc(X_res, y, cls_idx, prs_res, rrs_res, layer,
                                   feature, seed=s, v_indep=v_indep)
                          for s in range(N_REPEATS)])
        null = []
        for p in range(N_PERM):
            yp = rng.permutation(y)
            a = cv_auroc(X_res, yp, cls_idx, prs_res, rrs_res, layer, feature,
                         seed=p % N_REPEATS, v_indep=v_indep)
            if not np.isnan(a):
                null.append(a)
        null = np.array(null)
        pval = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        res[feature] = dict(auroc=float(obs), perm_mean=float(null.mean()),
                            perm_p95=float(np.percentile(null, 95)), p_value=pval,
                            n_perm=len(null))
        print(f"  {name:22} {feature:11} AUROC={obs:.4f}  "
              f"null={null.mean():.4f} (95th {np.percentile(null,95):.4f})  p={pval:.4f}")
    res["full_minus_baseline"] = res["full"]["auroc"] - res["baseline"]["auroc"]
    res["orth_indep_minus_baseline"] = res["orth_indep"]["auroc"] - res["baseline"]["auroc"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--prs", required=True)
    ap.add_argument("--rrs", required=True)
    ap.add_argument("--residual-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    E = load_extractor()
    stoi = pickle.load(open(args.meta, "rb"))["stoi"]; tok1 = stoi["1"]
    prs_lines, rrs_lines = E.read_lines(args.prs), E.read_lines(args.rrs)

    model, margs, iter_num = E.load_model(args.ckpt, args.device)
    P_prs = score(model, prs_lines, stoi, tok1, margs["block_size"], args.device)
    P_rrs = score(model, rrs_lines, stoi, tok1, margs["block_size"], args.device)
    del model; torch.cuda.empty_cache()

    prs_res = torch.load(Path(args.residual_dir) / "prs_residuals.pt").numpy()
    rrs_res = torch.load(Path(args.residual_dir) / "rrs_residuals.pt").numpy()
    L = args.layer
    thr = args.threshold

    n_fn = int((P_prs < thr).sum()); n_fp = int((P_rrs > thr).sum())
    print(f"model iter={iter_num}  layer={L}  threshold={thr}")
    print(f"PRS n={len(P_prs)}  RRS n={len(P_rrs)}   FN={n_fn}  FP={n_fp}")

    # ---- step 1: save the decomposition -------------------------------
    v_all = prs_res[:, L, :].mean(0) - rrs_res[:, L, :].mean(0)
    v_all = v_all / np.linalg.norm(v_all)
    proj_prs = prs_res[:, L, :] @ v_all
    proj_rrs = rrs_res[:, L, :] @ v_all
    perp_prs = prs_res[:, L, :] - np.outer(proj_prs, v_all)
    perp_rrs = rrs_res[:, L, :] - np.outer(proj_rrs, v_all)
    np.savez_compressed(
        out / f"decomposition_layer{L:02d}.npz",
        v=v_all, proj_prs=proj_prs, proj_rrs=proj_rrs,
        perp_norm_prs=np.linalg.norm(perp_prs, axis=1),
        perp_norm_rrs=np.linalg.norm(perp_rrs, axis=1),
        P_prs=P_prs, P_rrs=P_rrs)
    print(f"wrote {out}/decomposition_layer{L:02d}.npz")

    # ---- step 2: conditional error detection ---------------------------
    print("\nout-of-fold AUROC for detecting the ERROR class "
          f"({N_REPEATS} CV repeats, {N_PERM}-permutation null, pipeline pre-specified)")
    rng = np.random.default_rng(0)
    tasks = {}

    # Task A: among predicted-POSITIVE, find false positives (RRS)
    idxP = [(0, i) for i in range(len(P_prs)) if P_prs[i] > thr] + \
           [(1, i) for i in range(len(P_rrs)) if P_rrs[i] > thr]
    XA = np.vstack([prs_res[i, L, :] if c == 0 else rrs_res[i, L, :] for c, i in idxP])
    yA = np.array([1 if c == 1 else 0 for c, _ in idxP])   # 1 = false positive
    print(f"\nTask A  called INTERACTING: n={len(yA)}  (TP={int((yA==0).sum())}, FP={int(yA.sum())})")
    # v estimated from the DISJOINT set (the predicted-negatives) so that projecting
    # it out cannot exactly cancel this task's class-mean difference.
    negP = [i for i in range(len(P_prs)) if P_prs[i] <= thr]
    negR = [i for i in range(len(P_rrs)) if P_rrs[i] <= thr]
    vA = prs_res[negP, L, :].mean(0) - rrs_res[negR, L, :].mean(0)
    vA = vA / np.linalg.norm(vA)
    tasks["A_false_positive"] = run_task("A false-positive", XA, yA, idxP,
                                         prs_res, rrs_res, L, rng, v_indep=vA)

    # Task B: among predicted-NEGATIVE, find false negatives (PRS)
    idxN = [(0, i) for i in range(len(P_prs)) if P_prs[i] <= thr] + \
           [(1, i) for i in range(len(P_rrs)) if P_rrs[i] <= thr]
    XB = np.vstack([prs_res[i, L, :] if c == 0 else rrs_res[i, L, :] for c, i in idxN])
    yB = np.array([1 if c == 0 else 0 for c, _ in idxN])   # 1 = false negative
    print(f"\nTask B  called NON-INTERACTING: n={len(yB)}  (TN={int((yB==0).sum())}, FN={int(yB.sum())})")
    posP = [i for i in range(len(P_prs)) if P_prs[i] > thr]
    posR = [i for i in range(len(P_rrs)) if P_rrs[i] > thr]
    vB = prs_res[posP, L, :].mean(0) - rrs_res[posR, L, :].mean(0)
    vB = vB / np.linalg.norm(vB)
    tasks["B_false_negative"] = run_task("B false-negative", XB, yB, idxN,
                                         prs_res, rrs_res, L, rng, v_indep=vB)

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names = ["A: false positive\n(called interacting)", "B: false negative\n(called non-interacting)"]
    keys = ["A_false_positive", "B_false_negative"]
    w = 0.26
    xs = np.arange(2)
    series = [("baseline", "#3b4cc0", "baseline  [r·v]"),
              ("full", "#c1440e", "full residual (PCA)"),
              ("orth_indep", "#1b7837", "$r_\\perp$, v from disjoint set")]
    for j, (fk, col, lab) in enumerate(series):
        vals_ = [tasks[k][fk]["auroc"] for k in keys]
        p95 = [tasks[k][fk]["perm_p95"] for k in keys]
        off = (j - 1) * w
        ax[0].bar(xs + off, vals_, w, color=col, label=lab)
        for i in range(2):
            ax[0].plot([xs[i]+off-w/2, xs[i]+off+w/2], [p95[i]]*2, color="k", lw=1.4)
    ax[0].axhline(0.5, ls=":", color="0.5")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, fontsize=9)
    ax[0].set_ylabel("out-of-fold AUROC (error detection)")
    ax[0].set_ylim(0.2, 1.0); ax[0].grid(alpha=.25, axis="y")
    ax[0].legend(fontsize=8.5)
    ax[0].set_title("Error detection vs permutation 95th pct (black bars)", fontsize=11)

    ax[1].scatter(proj_prs, P_prs, s=16, c="darkorange", alpha=.7, label="PRS")
    ax[1].scatter(proj_rrs, P_rrs, s=16, c="royalblue", alpha=.7, label="RRS")
    ax[1].axhline(thr, ls=":", color="0.5")
    ax[1].set_xlabel("projection on class direction  $r\\cdot v$  (layer %d)" % L)
    ax[1].set_ylabel("P(interacting)")
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=9)
    ax[1].set_title("Decision variable vs model output", fontsize=11)

    fig.suptitle(f"iter{iter_num} — is model error detectable "
                 f"off the decision axis?", fontsize=12.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ppigplm_error_signature.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'ppigplm_error_signature.png'}")

    json.dump({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()), "iter_num": iter_num,
        "layer": L, "threshold": thr,
        "n_prs": len(P_prs), "n_rrs": len(P_rrs), "n_false_negative": n_fn,
        "n_false_positive": n_fp,
        "pipeline": {"pca_components": N_PCA, "logreg_C": C_REG,
                     "cv": "stratified 5-fold", "repeats": N_REPEATS,
                     "n_permutations": N_PERM,
                     "note": "hyperparameters pre-specified, no tuning; "
                             "scaler/PCA fit inside folds; v recomputed per fold"},
        "tasks": tasks,
    }, open(out / "error_signature_metrics.json", "w"), indent=2)
    print(f"wrote {out/'error_signature_metrics.json'}")


if __name__ == "__main__":
    main()
