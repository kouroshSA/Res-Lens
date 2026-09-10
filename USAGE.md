# USAGE

Run order, arguments, and the mistakes that cost time. Written from one full study; expect
to edit paths and module names for your model.

---

## 0. Prerequisites

**Environment**

```
python >= 3.10
torch  numpy  scikit-learn  scipy  matplotlib  pacmap
```

A GPU is recommended, not required. Reference: Python 3.12, torch 2.13+cu130,
scikit-learn 1.8, pacmap 0.8, on a 24 GB card.

**Model**

The scripts import your model class (e.g. `GPT`/`GPTConfig` from a nanoGPT-style
`model.py`). Adjust the `sys.path.insert` at the top of `residual_extract.py`.

They assume:

- a decoder-only transformer whose blocks write to the residual stream through two
  projections (`attn.c_proj`, `mlp.c_proj` in nanoGPT; `o_proj` / `down_proj` in
  Llama-style models)
- a fixed **readout position** whose logits produce the decision
- two classes of input, one sequence per line, in two text files

**Checkpoint selection**

Aggregate scripts read a CSV with `model` and `pick_iteration` columns.

> **Do not assume the final checkpoint is the right one.** Choose by an explicit,
> documented criterion and record the picks. In our study the last checkpoint was optimal
> for only one replicate out of ten — using it everywhere shifted every number.

---

## 1. Per-model extraction and geometry

```bash
python residual_extract.py \
  --ckpt   <ckpt>.pt \
  --meta   <tokenizer meta>.pkl \
  --prs    <class_A>.csv \
  --rrs    <class_B>.csv \
  --outdir out/<model>_iter<it> \
  --device cuda:0 \
  [--expect-auroc <recorded value>]
```

Writes `prs_residuals.pt` / `rrs_residuals.pt` of shape `(n, n_layers+1, d_model)`.
**Keep these** — every later step reuses them, and only the ablation steps need the GPU
again.

**`--expect-auroc` is a validation gate.** Pass any independently recorded performance
figure for that checkpoint. The script recomputes it from its own forward passes and warns
if it deviates by more than 0.05. Because that figure depends on the encoding, the
truncation and the readout position all being simultaneously correct, an exact match is
strong evidence the extraction is sound. Use it wherever such a reference exists; where it
does not, report your numbers as new measurements rather than reproductions.

Repeat per model, then:

```bash
python aggregate_geometry.py --root out --picks checkpoint_picks.csv --outdir out/aggregate
```

---

## 2. Causal ablation

```bash
python ablate.py \
  --ckpt <ckpt>.pt --meta <meta>.pkl \
  --prs <class_A>.csv --rrs <class_B>.csv \
  --residual-dir out/<model>_iter<it> \
  --outdir abl/<model>_iter<it> \
  --lambdas "0,0.5,0.75,0.9,1.0" --n-random 5

python aggregate_ablation.py --root abl --picks checkpoint_picks.csv --outdir abl/aggregate
```

> **The random-direction control is the result, not decoration.** Ablating anything degrades
> a model. The claim rests entirely on random directions of the same magnitude leaving
> performance unchanged. **If you shorten this run, do not drop `--n-random`.**

The RRS-only PC1 control is also reported, but it is generally **not independent** of `v`
(they can be highly aligned). Read it as dose-response, not as a second control.

---

## 3. Split-half control

```bash
python splithalf.py \
  --root <checkpoints root> --residual-root out \
  --picks checkpoint_picks.csv --outdir splithalf \
  --n-splits 10 --n-random 5
```

Estimates `v` on one half, ablates, evaluates on the other; 20 held-out evaluations per
model. **Report this gap, not the in-sample one from `ablate.py`** — estimating the
direction on half the data makes it noisier, so the in-sample figure is optimistic.

---

## 4. Cross-model projection

```bash
python pooled_projection.py --residual-root out --picks checkpoint_picks.csv --outdir pooled
python figures_combined.py  --residual-root out --picks checkpoint_picks.csv --outdir combined --seed 0
```

`pooled_projection.py` reports silhouette **by class** and **by model identity** for each
layer. Whichever is larger is what the picture is actually showing — if model identity
dominates, the panel says nothing about class structure.

---

## 5. Optional — error margin

```bash
python error_margin.py --ckpt <ckpt>.pt --meta <meta>.pkl \
  --prs <class_A>.csv --rrs <class_B>.csv \
  --residual-dir out/<model>_iter<it> --outdir err
```

Asks whether model error is predictable from residual geometry **beyond the decision
variable itself**. Read the permutation p-value, not the raw AUROC.

---

## 6. Sanity checks

Two things should hold regardless of model:

- the **`--expect-auroc` gate** passes wherever you have a reference
- the **random-direction control** returns ~baseline

If `|cos(v, PC1)|` does not rise toward the decision layer, or the random control is not
~baseline, check the readout position and the vocabulary before interpreting anything.

---

## 7. Pitfalls

**Wrong checkpoint.** See §0. Costs a full re-run.

**Wrong evaluation set or tokenizer generation.** If a project has more than one tokenizer,
sets built for one are silently incompatible with the other. **Check the delimiter first:**
`rfind(delim)` returns −1 when absent, so a vocabulary check over `line[:rfind+1]` passes
*vacuously* on a file in a different format, and the run dies later inside the embedding
with an opaque dtype error (an empty id list makes `torch.tensor([[]])` a float tensor).
`residual_extract.py` checks the delimiter first and fails with a clear message.

**Scope.** Only run where the model actually discriminates. If baseline performance is near
chance on an evaluation set, there is no class direction to find and the geometry is noise.

**Silhouette is the wrong instrument for dimensionality.** It uses distance across all
`d_model` dimensions, so a clean split along one axis is diluted by spread in the rest. A
low silhouette alongside `|cos(v, PC1)| ≈ 1` is not a contradiction. Use the
variance-fraction and PC1 columns for that question.

**Do not project out `v` and then ask a classifier to recover the class.** `v` *is* the
class-mean difference, so removing it zeroes the training class-mean difference exactly, the
logistic solver terminates at `n_iter = 0`, and AUROC comes back as **exactly 0.5000** — an
artifact of construction, not a null result. The suspiciously round number is the tell.
`error_margin.py` documents this and estimates `v` from disjoint samples instead.

**PaCMAP is stochastic.** Unseeded runs of identical data can give visibly different
embeddings and silhouettes. Pass a seed. Treat embedding silhouettes as illustrative; the
full-dimensional measures are the stable ones.

**Cross-model pooling needs care.** Independently trained models have unaligned spaces.
Z-scoring per model removes location and scale but **not rotation**. A "model dominates"
pattern in early layers is partly just that unremoved misalignment and must not be read as
absence of shared structure.

**Check what is holding your GPU.** Two ablation runs failed here with OOM that looked like
a model-size problem. The actual cause was an unrelated server process holding most of both
cards. `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` first.

---

## 8. Reporting

Report the **held-out** specificity gap, not the in-sample one.

If the random-direction control is not clean, report that instead of the headline number.
A null result — "no single direction carries this decision" — is a real finding and should
be reported as one.
