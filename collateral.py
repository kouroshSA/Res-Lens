#!/usr/bin/env python3
"""
Collateral-damage control: does ablating v break the DECISION, or break the MODEL?

Generated: 2026-09-11 09:00:00

THE QUESTION THIS SETTLES
  Ablating the class direction collapses PRS-vs-RRS discrimination. A fair objection is
  that this is close to circular: v is defined as the class-mean difference, so removing
  it and finding the classes no longer separate has a tautological flavour.

  The split-half control already shows the effect is not an artifact of fitting v to the
  evaluated pairs. This adds the complementary control, the one Heretic uses for language
  models (KL divergence from the original model on unrelated prompts) and which was
  missing here:

      is the model OTHERWISE INTACT after the edit?

  If ablating v destroys the interaction decision while leaving ordinary next-token
  sequence modelling essentially unchanged, the direction is specifically the decision
  axis -- specificity of FUNCTION, not merely of direction, which is much harder to call
  circular. If general modelling degrades comparably, then the edit broke the model and
  the specificity claim weakens.

WHAT IS MEASURED
  On the same sequences, three things per condition:

    task AUROC        class A vs class B from P(class token) at the decision position
    general CE        mean next-token cross-entropy over ALL ordinary positions
                      (the decision position excluded) -- i.e. can the model still
                      predict amino acids?
    general KL        mean KL(original || ablated) of the next-token distribution at
                      those same ordinary positions -- the direct Heretic analogue

  Conditions: baseline, ablate v, ablate N random unit directions (same magnitude).

  The decisive comparison is NOT "is general CE unchanged" -- any edit perturbs it a
  little. It is whether ablating v costs much more GENERAL damage than ablating a random
  direction, relative to how much more TASK damage it causes.

Usage:
  python ppigplm_collateral_20260911_090000.py --root <ckpt root> \
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
MODELS = [f"V3-{i}" for i in range(1, 11)]   # edit to your replicate ids


def load_extractor():
    spec = importlib.util.spec_from_file_location("residual_extract", EXTRACTOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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


@torch.no_grad()
def evaluate(model, lines, stoi, tok1, block_size, device, ref_logprobs=None):
    """Returns P(class token) per sequence, mean general CE, per-position logprobs.

    'General' positions are every next-token prediction EXCEPT the final one, which is
    the decision. So general CE measures ordinary sequence modelling only.
    """
    probs, ces, kls, store = [], [], [], []
    for k, line in enumerate(lines):
        i = line.rfind("<")
        ids = [stoi[c] for c in line[:i + 1]][-block_size:]
        x = torch.tensor([ids], dtype=torch.long, device=device)

        # decision readout: logits at the final position
        logits_last, _ = model(x)
        probs.append(F.softmax(logits_last[0, -1, :].float(), 0)[tok1].item())

        # general next-token modelling: needs logits at every position, which the
        # model only returns when targets are supplied
        inp, tgt = x[:, :-1], x[:, 1:]
        logits_all, loss = model(inp, tgt)
        ces.append(float(loss))

        lp = F.log_softmax(logits_all[0].float(), dim=-1)      # (T-1, vocab)
        if ref_logprobs is not None:
            ref = ref_logprobs[k].to(lp.device)
            n = min(ref.shape[0], lp.shape[0])
            # KL(original || ablated), averaged over ordinary positions
            kls.append(float(F.kl_div(lp[:n], ref[:n], reduction="batchmean",
                                      log_target=True)))
        else:
            store.append(lp.cpu())
    return np.array(probs), float(np.mean(ces)), (np.mean(kls) if kls else None), store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--residual-root", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=3)
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
        if not (rd / "a_residuals.pt").exists():
            print(f"[skip] {m}"); continue
        a_res = torch.load(rd / "a_residuals.pt").numpy()
        b_res = torch.load(rd / "b_residuals.pt").numpy()
        stoi = pickle.load(open(root / m / "meta.pkl", "rb"))["stoi"]
        tok1 = stoi["1"]
        a_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"PRS-{m}.csv")
        b_lines = E.read_lines(root / m / "eval_sets/PRS-RRS" / f"RRS-{m}.csv")
        lines = a_lines + b_lines
        y = [1] * len(a_lines) + [0] * len(b_lines)

        model, margs, _ = E.load_model(str(root / m / "checkpoints" / f"ckpt_{it}.pt"),
                                       args.device)
        bs, nembd = margs["block_size"], margs["n_embd"]
        snap = snapshot(model)

        # ---- baseline, capturing reference distributions ------------------
        p0, ce0, _, ref = evaluate(model, lines, stoi, tok1, bs, args.device)
        au0 = roc_auc_score(y, p0)

        v = a_res[:, L, :].mean(0) - b_res[:, L, :].mean(0)
        v = torch.from_numpy(v / np.linalg.norm(v)).float()

        restore(model, snap); ablate(model, v, 1.0)
        pv, cev, klv, _ = evaluate(model, lines, stoi, tok1, bs, args.device, ref)
        auv = roc_auc_score(y, pv)

        aur, cer, klr = [], [], []
        for s in range(args.n_random):
            g = torch.Generator().manual_seed(2000 + s)
            rv = torch.randn(nembd, generator=g); rv = rv / rv.norm()
            restore(model, snap); ablate(model, rv, 1.0)
            pr, cerr, klrr, _ = evaluate(model, lines, stoi, tok1, bs, args.device, ref)
            aur.append(roc_auc_score(y, pr)); cer.append(cerr); klr.append(klrr)

        restore(model, snap); del model; torch.cuda.empty_cache()

        row = dict(model=m, ckpt=it,
                   auroc_base=au0, auroc_v=float(auv), auroc_rand=float(np.mean(aur)),
                   ce_base=ce0, ce_v=cev, ce_rand=float(np.mean(cer)),
                   kl_v=float(klv), kl_rand=float(np.mean(klr)))
        row["task_damage_v"] = au0 - row["auroc_v"]
        row["task_damage_rand"] = au0 - row["auroc_rand"]
        row["general_damage_v"] = cev - ce0
        row["general_damage_rand"] = row["ce_rand"] - ce0
        rows.append(row)
        print(f"{m:6} ckpt_{it:<5} AUROC {au0:.4f}->{row['auroc_v']:.4f} (rand {row['auroc_rand']:.4f}) | "
              f"CE {ce0:.4f}->{cev:.4f} (rand {row['ce_rand']:.4f}) | "
              f"KL v={klv:.5f} rand={np.mean(klr):.5f}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows], float)

    print("\n" + "=" * 78)
    print(f"across {len(rows)} models (mean +/- sd)")
    print(f"  task damage   ablate v    : {col('task_damage_v').mean():+.4f} +/- {col('task_damage_v').std(ddof=1):.4f}")
    print(f"  task damage   random      : {col('task_damage_rand').mean():+.4f} +/- {col('task_damage_rand').std(ddof=1):.4f}")
    print(f"  general CE +  ablate v    : {col('general_damage_v').mean():+.4f} +/- {col('general_damage_v').std(ddof=1):.4f}")
    print(f"  general CE +  random      : {col('general_damage_rand').mean():+.4f} +/- {col('general_damage_rand').std(ddof=1):.4f}")
    print(f"  general KL    ablate v    : {col('kl_v').mean():.5f} +/- {col('kl_v').std(ddof=1):.5f}")
    print(f"  general KL    random      : {col('kl_rand').mean():.5f} +/- {col('kl_rand').std(ddof=1):.5f}")

    # the decisive ratio: task damage per unit of general damage
    eps = 1e-9
    sel_v = col("task_damage_v") / (col("kl_v") + eps)
    sel_r = col("task_damage_rand") / (col("kl_rand") + eps)
    print(f"\n  selectivity (task damage per unit general KL)")
    print(f"    ablate v : {sel_v.mean():.1f}")
    print(f"    random   : {sel_r.mean():.1f}")
    print("\n  A large ratio for v and a small one for random means the edit is surgical:")
    print("  it removes the decision without breaking general sequence modelling.")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    x = np.arange(len(rows)); w = 0.35
    ax[0].bar(x - w/2, col("task_damage_v"), w, color="#c1440e", label="ablate $v$")
    ax[0].bar(x + w/2, col("task_damage_rand"), w, color="#3b4cc0", label="random direction")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r["model"] for r in rows], fontsize=8, rotation=45)
    ax[0].set_ylabel("AUROC lost"); ax[0].set_title("TASK damage", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25, axis="y")

    ax[1].bar(x - w/2, col("kl_v"), w, color="#c1440e", label="ablate $v$")
    ax[1].bar(x + w/2, col("kl_rand"), w, color="#3b4cc0", label="random direction")
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["model"] for r in rows], fontsize=8, rotation=45)
    ax[1].set_ylabel("KL(original ‖ ablated), ordinary positions")
    ax[1].set_title("GENERAL damage — next-token modelling", fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25, axis="y")

    fig.suptitle("Does ablating the class direction break the DECISION or the MODEL?",
                 fontsize=12.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"collateral_damage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out/'collateral_damage.png'}")

    with open(out / "collateral_summary.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script": str(Path(__file__).resolve()), "layer": L,
               "n_random": args.n_random, "picks": picks, "per_model": rows,
               "across_model": {k: dict(mean=float(col(k).mean()), sd=float(col(k).std(ddof=1)))
                                for k in ("task_damage_v", "task_damage_rand",
                                          "general_damage_v", "general_damage_rand",
                                          "kl_v", "kl_rand")},
               "selectivity_v": float(sel_v.mean()),
               "selectivity_random": float(sel_r.mean())},
              open(out / "collateral_summary.json", "w"), indent=2)
    print(f"wrote {out/'collateral_summary.csv'} and .json")


if __name__ == "__main__":
    main()
