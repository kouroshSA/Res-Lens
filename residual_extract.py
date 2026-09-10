#!/usr/bin/env python3
"""
Residual-stream extractor + geometry analysis for nanoGPT-style models.

Generated: 2026-09-05 10:00:00

This is the adaptation of directional-ablation analysis to nanoGPT. It asks: is the
model's PRS-vs-RRS discrimination carried by a LINEAR direction in the residual
stream, and if so, at which layer does it emerge?

CONTRAST
  PRS (positive reference set, true interacting pairs) <-> "harmful" prompts in Heretic
  RRS (random reference set, non-interacting)        <-> "harmless" prompts
  direction v_L = normalize(mean_resid_PRS - mean_resid_RRS) at each layer L

EXTRACTION
  nanoGPT's GPT has no output_hidden_states, so residuals are captured with forward
  hooks (heretic gets these from HF's generate(output_hidden_states=True)):
    layer 0      = transformer.drop output   == drop(wte(idx) + wpe(pos))
    layer 1..n   = transformer.h[i] output
  matching Heretic's convention where index 0 is the embedding output.
  Residual is taken at the LAST position, which is the position whose logits
  produce the classification token -- see READOUT below.

READOUT (must match the project's existing eval exactly, live_eval_geom.py:score_all)
    i = line.rfind("<")
    ids = [stoi[c] for c in line[:i+1]][-block_size:]
    logits, _ = model(tensor([ids]))
    P = softmax(logits[0, -1, :])[stoi["1"]]
  GPT.forward with targets=None returns logits ONLY for the final position, so
  logits[0,-1,:] is that position.

VALIDATION GATE
  The script recomputes AUROC from those probabilities and compares it to a value
  you supply via --expect-auroc. If the encoding, truncation, or readout position
  were wrong, AUROC would collapse toward 0.5 and the geometry would be measuring
  noise. Do not trust the geometry unless this gate passes.

  Reference (from this repo's own logs, the evaluation set):
  a recorded reference value, if one exists.
  On the OOD and E.coli eval sets these same models sit at 0.46-0.59, i.e. chance --
  there is no interaction direction to find there, so running this on those sets
  is not meaningful.

Every number written out is computed from the loaded checkpoint and input files.

Usage:
  python residual_extract.py \
      --ckpt <ckpt>.pt \
      --meta <meta>.pkl \
      --prs <class_A>.csv \
      --rrs <class_B>.csv \
      --outdir residual_analysis_<TS> --expect-auroc 0.84
"""

import argparse
import json
import pickle
import sys
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
from sklearn.metrics import roc_auc_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402

PRS_COLOR = "darkorange"   # interacting  (signal class)
RRS_COLOR = "royalblue"    # non-interacting


def load_model(ckpt_path, device):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = c["model_args"]
    m = GPT(GPTConfig(**args))
    sd = c["model"]
    for k in list(sd):                      # strip torch.compile prefix
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m, args, c.get("iter_num")


def read_lines(path):
    return [l.rstrip("\n") for l in open(path) if l.strip()]


class ResidualCatcher:
    """Forward hooks capturing the residual stream at every layer boundary.

    nanoGPT Block.forward is x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)),
    so a block's OUTPUT is the residual stream after that block.
    """

    def __init__(self, model):
        self.model = model
        self.buf = {}
        self.handles = []

    def __enter__(self):
        def mk(idx):
            def hook(_mod, _inp, out):
                # out: (b, t, n_embd) -> keep last position only
                self.buf[idx] = out[:, -1, :].detach().float().cpu()
            return hook

        # layer 0 = embedding output (post-dropout), matching heretic's convention
        self.handles.append(self.model.transformer.drop.register_forward_hook(mk(0)))
        for i, block in enumerate(self.model.transformer.h):
            self.handles.append(block.register_forward_hook(mk(i + 1)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self, n_layers):
        return torch.cat([self.buf[i] for i in range(n_layers + 1)], dim=0)


@torch.no_grad()
def extract(model, lines, stoi, tok1, block_size, n_layers, device, tag):
    """Returns (residuals (n_seq, n_layers+1, n_embd) float32, probs (n_seq,))."""
    res, probs = [], []
    with ResidualCatcher(model) as catcher:
        for k, line in enumerate(lines):
            i = line.rfind("<")
            ids = [stoi[c] for c in line[:i + 1]][-block_size:]
            x = torch.tensor([ids], device=device)
            logits, _ = model(x)
            probs.append(F.softmax(logits[0, -1, :].float(), 0)[tok1].item())
            res.append(catcher.stacked(n_layers))     # (n_layers+1, n_embd)
            if (k + 1) % 25 == 0:
                print(f"[{tag}] {k+1}/{len(lines)}", flush=True)
    return torch.stack(res, dim=0), np.array(probs)


def silhouette(a, b, layer):
    x = np.vstack((a[:, layer, :].numpy(), b[:, layer, :].numpy()))
    y = np.concatenate((np.zeros(len(a)), np.ones(len(b))))
    return float(silhouette_score(x, y))


def class_mean_cosine(a, b, layer):
    ma = a[:, layer, :].mean(dim=0)
    mb = b[:, layer, :].mean(dim=0)
    return float(F.cosine_similarity(ma, mb, dim=-1))


def direction_norm(a, b, layer):
    """|mean_PRS - mean_RRS| relative to the mean residual norm at that layer."""
    ma = a[:, layer, :].mean(dim=0)
    mb = b[:, layer, :].mean(dim=0)
    scale = 0.5 * (ma.norm() + mb.norm())
    return float((ma - mb).norm() / scale) if scale > 0 else 0.0


def pca_alignment(a, b, layer):
    """Where does the class direction sit in the variance spectrum?

    Silhouette uses full-space Euclidean distance, so it is a POOR instrument for
    detecting a single dominant separating axis: a large between-class gap along one
    direction gets diluted by within-class scatter spread over the other ~199
    dimensions. This function measures the thing silhouette cannot see -- whether the
    class-mean-difference direction v is itself a principal axis of the data.

    Returns:
      var_frac  fraction of total variance the data carries along v
      cos_pc1   |cos(v, PC1)|; ~1.0 means v IS the dominant variance axis
      rank      how many PCs carry more variance than v (1 = v is the top axis)
    """
    x = np.vstack((a[:, layer, :].numpy(), b[:, layer, :].numpy()))
    xc = x - x.mean(0)
    v = a[:, layer, :].numpy().mean(0) - b[:, layer, :].numpy().mean(0)
    n = np.linalg.norm(v)
    if n == 0:
        return 0.0, 0.0, len(xc)
    v = v / n
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2 / (len(x) - 1)
    var_v = float(np.var(xc @ v, ddof=1))
    return (float(var_v / var.sum()),
            float(abs(np.dot(v, vt[0]))),
            int((var > var_v).sum()) + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--prs", required=True)
    p.add_argument("--rrs", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expect-auroc", type=float, default=None,
                   help="Reference AUROC. Warns loudly if the computed value deviates by >0.05.")
    p.add_argument("--pacmap-layers", default="",
                   help="Comma-separated layers for PaCMAP panels. Default: 4 spread over depth.")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    meta = pickle.load(open(args.meta, "rb"))
    stoi = meta["stoi"]
    if "1" not in stoi:
        raise SystemExit('meta stoi has no "1" token -- wrong vocabulary for this eval set?')
    tok1 = stoi["1"]

    model, margs, iter_num = load_model(args.ckpt, device)
    n_layers = margs["n_layer"]
    block_size = margs["block_size"]
    print(f"model: {args.ckpt}")
    print(f"  n_layer={n_layers} n_embd={margs['n_embd']} vocab={margs['vocab_size']} "
          f"block_size={block_size} iter={iter_num}")

    prs_lines, rrs_lines = read_lines(args.prs), read_lines(args.rrs)
    for name, lines in (("PRS", prs_lines), ("RRS", rrs_lines)):
        # The '<' delimiter check MUST come first. rfind returns -1 when absent,
        # so l[:rfind+1] is the empty string and a vocabulary check over it passes
        # vacuously -- which silently admits files in a different prompt format
        # (e.g. the ppiDCE/ppiBTEP sets, which carry no '<' at all) and then dies
        # later inside the embedding with an opaque dtype error.
        missing = [i for i, l in enumerate(lines) if "<" not in l]
        if missing:
            raise SystemExit(
                f"{name}: {len(missing)} of {len(lines)} lines contain no '<' delimiter "
                f"(first at line {missing[0]+1}). This file is not in the expected prompt format "
                f"(expected '...,<'). Wrong eval set for this model."
            )
        bad = {c for l in lines for c in l[:l.rfind('<') + 1] if c not in stoi}
        if bad:
            raise SystemExit(f"{name} contains characters absent from the vocabulary: {sorted(bad)}")
    print(f"PRS: {len(prs_lines)} sequences   RRS: {len(rrs_lines)} sequences")

    prs_res, prs_p = extract(model, prs_lines, stoi, tok1, block_size, n_layers, device, "PRS")
    rrs_res, rrs_p = extract(model, rrs_lines, stoi, tok1, block_size, n_layers, device, "RRS")
    print(f"residual shapes: PRS={tuple(prs_res.shape)} RRS={tuple(rrs_res.shape)}")

    # ---- validation gate -------------------------------------------------
    auroc = roc_auc_score([1] * len(prs_p) + [0] * len(rrs_p),
                          np.concatenate([prs_p, rrs_p]))
    print(f"\nAUROC (recomputed from this script's own forward passes): {auroc:.4f}")
    print(f"  PRS median P(1) = {np.median(prs_p):.4f}   RRS median P(1) = {np.median(rrs_p):.4f}")
    gate = "not checked"
    if args.expect_auroc is not None:
        delta = abs(auroc - args.expect_auroc)
        gate = "PASS" if delta <= 0.05 else "FAIL"
        print(f"  expected ~{args.expect_auroc:.4f}  |delta|={delta:.4f}  -> {gate}")
        if gate == "FAIL":
            print("  [!] Encoding/readout may be wrong. Geometry below is NOT trustworthy.")

    torch.save(prs_res, out / "prs_residuals.pt")
    torch.save(rrs_res, out / "rrs_residuals.pt")

    # ---- geometry --------------------------------------------------------
    layers = list(range(n_layers + 1))
    sil = [silhouette(prs_res, rrs_res, L) for L in layers]
    cos = [class_mean_cosine(prs_res, rrs_res, L) for L in layers]
    dn = [direction_norm(prs_res, rrs_res, L) for L in layers]
    align = [pca_alignment(prs_res, rrs_res, L) for L in layers]
    var_frac = [a[0] for a in align]
    cos_pc1 = [a[1] for a in align]
    pc_rank = [a[2] for a in align]

    print("\nlayer  silhouette  cos(means)  |dmean|/|mean|   var_frac(v)  |cos(v,PC1)|  PCrank")
    for L in layers:
        print(f"{L:5d}  {sil[L]:10.4f}  {cos[L]:10.4f}  {dn[L]:14.4f}  "
              f"{var_frac[L]:11.4f}  {cos_pc1[L]:12.3f}  {pc_rank[L]:6d}")
    best = int(np.argmax(sil))
    print(f"\npeak separation at layer {best} (silhouette {sil[best]:.4f})")
    print(f"at final layer: v carries {100*var_frac[-1]:.1f}% of variance, "
          f"|cos(v,PC1)|={cos_pc1[-1]:.3f}, variance rank {pc_rank[-1]}")

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.4), constrained_layout=True)
    ax[0].plot(layers, sil, "-o", ms=4, color="#c1440e")
    ax[0].axhline(0, lw=.8, ls="--", color="#999")
    ax[0].set_xlabel("Layer"); ax[0].set_ylabel("Silhouette score")
    ax[0].set_title("PRS vs RRS separation (full residual space)", fontsize=11)
    ax[1].plot(layers, cos, "-o", ms=4, color="#1b7837")
    ax[1].set_xlabel("Layer"); ax[1].set_ylabel("cos(mean PRS, mean RRS)")
    ax[1].set_title("Class-mean alignment (1.0 = indistinguishable)", fontsize=11)
    ax[2].plot(layers, dn, "-o", ms=4, color="#3b4cc0")
    ax[2].set_xlabel("Layer"); ax[2].set_ylabel("|mean PRS - mean RRS| / |mean|")
    ax[2].set_title("Relative class-mean separation", fontsize=11)
    ax[3].plot(layers, cos_pc1, "-o", ms=4, color="#7b3294", label="|cos(v, PC1)|")
    ax[3].plot(layers, var_frac, "-s", ms=4, color="#c1440e", label="variance fraction along v")
    ax[3].set_xlabel("Layer"); ax[3].set_ylabel("alignment / variance fraction")
    ax[3].set_title("Class direction vs principal axis", fontsize=11)
    ax[3].set_ylim(0, 1.02)
    ax[3].legend(fontsize=8.5, loc="upper left")
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle(
        f"residual geometry  |  {Path(args.ckpt).parent.name} (iter {iter_num})  |  "
        f"AUROC {auroc:.3f}  |  PRS n={len(prs_lines)}, RRS n={len(rrs_lines)}",
        fontsize=12.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ppigplm_residual_geometry.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'ppigplm_residual_geometry.png'}")

    # ---- PaCMAP panels ---------------------------------------------------
    try:
        from pacmap import PaCMAP
        sel = ([int(x) for x in args.pacmap_layers.split(",") if x.strip()]
               if args.pacmap_layers.strip()
               else sorted({0, n_layers // 3, 2 * n_layers // 3, n_layers}))
        n_pts = len(prs_lines) + len(rrs_lines)
        nn = max(5, min(30, n_pts // 6))          # keep n_neighbors sane for small n
        print(f"PaCMAP layers {sel}, n_neighbors={nn} (n={n_pts})")
        fig, axes = plt.subplots(1, len(sel), figsize=(4.4 * len(sel), 4.7),
                                 constrained_layout=True)
        if len(sel) == 1:
            axes = [axes]
        init = None
        for ax_, L in zip(axes, sel):
            X = np.vstack((prs_res[:, L, :].numpy(), rrs_res[:, L, :].numpy()))
            emb = PaCMAP(n_components=2, n_neighbors=nn).fit_transform(X, init=init)
            init = emb
            a, b = emb[:len(prs_lines)], emb[len(prs_lines):]
            ax_.scatter(b[:, 0], b[:, 1], s=14, c=RRS_COLOR, alpha=.6,
                        label="RRS (non-interacting)", edgecolors="none")
            ax_.scatter(a[:, 0], a[:, 1], s=14, c=PRS_COLOR, alpha=.6,
                        label="PRS (interacting)", edgecolors="none")
            ax_.set_title(f"Layer {L:02d}   silhouette {sil[L]:.3f}", fontsize=11)
            ax_.set_xticks([]); ax_.set_yticks([])
        axes[-1].legend(loc="best", fontsize=8.5, framealpha=.9)
        fig.suptitle("residual vectors, PaCMAP projection", fontsize=12.5)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"ppigplm_residual_pacmap.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out/'ppigplm_residual_pacmap.png'}")
    except ImportError:
        print("pacmap not available -- skipped projection panels")

    json.dump({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()), "iter_num": iter_num,
        "meta": str(Path(args.meta).resolve()),
        "prs": str(Path(args.prs).resolve()), "rrs": str(Path(args.rrs).resolve()),
        "model_args": margs,
        "n_prs": len(prs_lines), "n_rrs": len(rrs_lines),
        "auroc": auroc, "expect_auroc": args.expect_auroc, "validation_gate": gate,
        "prs_median_P1": float(np.median(prs_p)), "rrs_median_P1": float(np.median(rrs_p)),
        "silhouette_by_layer": sil, "class_mean_cosine_by_layer": cos,
        "relative_mean_separation_by_layer": dn,
        "variance_fraction_along_v_by_layer": var_frac,
        "abs_cos_v_pc1_by_layer": cos_pc1,
        "variance_rank_of_v_by_layer": pc_rank,
        "peak_separation_layer": best,
    }, open(out / "metrics.json", "w"), indent=2)
    print(f"wrote {out/'metrics.json'}")


if __name__ == "__main__":
    main()
