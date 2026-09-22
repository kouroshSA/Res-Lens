# Res-Lens

**A lens on the residual stream — and a scalpel for it.**

Two things in one toolkit:

- **the lens** — where in a transformer's residual stream a binary decision forms, and
  whether the separating direction is a principal axis of the representation
- **the scalpel** — whether that decision *causally depends* on that direction, tested by
  editing it out of the weights and measuring what breaks

The second is the point. The first describes where things sit; only an intervention shows
dependence.

Written for character-level [nanoGPT](https://github.com/karpathy/nanoGPT)-style models that
classify pairs of sequences, but the method applies to any decoder-only transformer with a
two-class contrast and a fixed readout position.

---

## The questions these tools answer

A transformer maintains a running internal summary as it reads — the **residual stream**,
updated by each block. If a model separates two classes, that separation must appear
somewhere in that summary.

1. **Where** does it appear, layer by layer? *(descriptive)*
2. **Is the separating direction a principal axis** of the representation, or a minor
   feature? *(descriptive)*
3. **Does discrimination causally depend on it** — does deleting it break the model?
   *(causal)*
4. **Does that survive when the direction is estimated on held-out data?** *(control)*
5. **Do independently trained replicates converge on the same geometry?** *(cross-model)*

Question 3 is the point. Questions 1–2 describe where things sit; only an intervention shows
dependence.

---

## Two spaces, two interventions — do not conflate them

The direction `v` and the *edit* live in different places, and the toolkit does both kinds
of edit. This trips people up, so it is stated explicitly:

| | where `v` comes from | what the intervention changes | scripts |
|---|---|---|---|
| **ablation** | activations | **model weights**, permanently | `ablate.py`, `splithalf.py`, `collateral.py` |
| **steering** | activations | **activations at run time**, weights untouched | `steer.py` |

**`v` is always derived from activations.** Run the model over many class-A inputs and
many class-B inputs, average the residual stream at the readout position, and subtract:

```
v = normalize( mean(class A) − mean(class B) )
```

Individual inputs average out; what survives is the axis along which the model separates the
two classes. `v` is *not* read off the weights, and it is *not* a token or an embedding —
nothing is removed from the vocabulary and **the embedding table is never modified**.

**Ablation edits weights.** Each block writes into the residual stream through projection
matrices. Replacing `W` with `(I − v vᵀ)W` leaves them able to compute everything else, but
their output has exactly zero component along `v`, so later layers never see the feature:

```
W  ←  W − λ · v (vᵀ W)          λ = 1  ⇒  (I − v vᵀ) W
```

No retraining, no gradients — one rank-1 edit per matrix, applied to every block. The change
is persistent: the modified model is a different model on disk. `splithalf.py` and
`collateral.py` snapshot the original weights and restore them between conditions.

**Steering edits activations.** `steer.py` adds `α·v` to the residual stream through forward
hooks, at the readout position only. **No weight is modified**; removing the hooks restores
the original model exactly.

```
h  ←  h + α · v
```

Because a *learned feature computed from the whole input* is removed (or added) rather than
a lexical item, the intervention generalises to inputs never used to compute `v` — which
`splithalf.py` tests directly. And because one direction out of `d_model` is touched,
collateral damage is small — which `collateral.py` measures.

---

## The control that makes it an experiment

**A drop in performance after ablation proves nothing on its own** — mutilating any model
degrades it. Every ablation script therefore also ablates **random unit directions of the
same magnitude** and reports:

```
specificity gap = performance(random direction) − performance(class direction)
```

A large positive gap means the collapse is specific to the class direction. A gap near zero
means you measured generic damage, and the honest move is to say so.

### The second control — specificity of *function*

There is a further objection the random-direction control does **not** answer: `v` is
*defined* as the class-mean difference, so removing it and finding the classes no longer
separate has a tautological flavour.

`collateral.py` addresses it. It measures, alongside task performance, whether **general
next-token modelling survives the edit** — `KL(original ‖ ablated)` at ordinary sequence
positions, with the decision position excluded.

A purely algebraic deletion of the separating axis carries **no prediction** about what
happens to everything else. If the task collapses while general modelling is essentially
untouched, the damage is confined to one *function* — which cannot be obtained by
construction. If general modelling degrades comparably, you broke the model and should say
so.

---

## Necessity and sufficiency

`ablate.py` removes `v` and asks whether discrimination dies — **necessity**.
`steer.py` adds `±αv` and asks whether the decision *moves* — **sufficiency**. Together
they are the standard causal pair; ablation alone gives only half.

Steering carries its own trap: a large `α` pins the output regardless of input, which is a
broken model, not control. `steer.py` therefore reports the operating point **and** AUROC at
every `α`, and computes its verdict over a **usable window** — the range where both the
random control and `v`'s own ranking stay near baseline. Expect the window to be narrow;
outside it, saturation.

## Scripts

| script | what it does |
|---|---|
| `residual_extract.py` | hooks the residual stream at every layer at the readout position; per-layer silhouette, class-mean cosine, variance fraction along `v`, \|cos(v, PC1)\|, PaCMAP panels |
| `ablate.py` | causal ablation — **edits weights**; random-direction and PC1 controls, λ sweep |
| `splithalf.py` | estimates `v` on one half, ablates (weights), evaluates on the other — circularity control |
| `aggregate_geometry.py` | cross-model layer curves and per-model tables |
| `aggregate_ablation.py` | cross-model ablation summary and specificity gaps |
| `pooled_projection.py` | pools replicates into one projection per layer; reports whether it separates by **class** or by **model identity** |
| `figures_combined.py` | combined class-only figures, light and dark |
| `error_margin.py` | is model error predictable from geometry, beyond the decision variable itself? |
| `collateral.py` | does the edit break the **decision** or the **model**? task damage vs general next-token KL |
| `steer.py` | adds `±αv` via forward hooks — **edits activations only**; sufficiency, with a saturation check |
| `selfcheck.py` | mechanical consistency checks — run after any edit, rename or port |

Each writes figures as **PNG + vector PDF** (300 dpi, except the multi-panel grids in
`pooled_projection.py` and `figures_combined.py`, which use 200/220 dpi to keep large grids
a sane file size), a **JSON metrics file** (the name varies per script —
`metrics.json`, `ablation_metrics.json`, `steering.json`, …), and where relevant a `.csv`.
Every plotted number is recomputed from the inputs.

**See [`USAGE.md`](USAGE.md)** for run order, arguments, sanity checks and a pitfalls
section.

---

## Adapting to your model

Class names in the code are **PRS** and **RRS** (positive / random reference set — standard
protein-interaction vocabulary). They are just labels for "class A" and "class B"; the method
is agnostic to what they mean.

### Porting checklist

Model-specific names appear in **five** files, not one. Editing only `ablate.py` leaves the
others silently wrong — `collateral.py` and `splithalf.py` carry their own copies of the
weight-edit and snapshot/restore helpers.

| file | what to change |
|---|---|
| `residual_extract.py` | `ResidualCatcher` — the hook targets (`transformer.drop`, `transformer.h[i]`) |
| `ablate.py` | `ablate_()` — the residual-writing projections (`block.attn.c_proj`, `block.mlp.c_proj`) |
| `splithalf.py` | its own `ablate()`, `snapshot()`, `restore()` — same projections |
| `collateral.py` | its own `ablate()`, `snapshot()`, `restore()` — same projections |
| `steer.py` | `Steerer` — the blocks it hooks (`transformer.h`) |

Also change the readout in each script's `score()` / `evaluate()` if your decision is not
"softmax at the final position, probability of one token".

For Llama-style models the projections are usually `self_attn.o_proj` and `mlp.down_proj`,
and the block list is `model.layers`.

A quick check that you got them all:

```bash
python selfcheck.py
```

`selfcheck.py` verifies the invariants that otherwise break **silently** — chiefly that the
residual filenames the extractor writes are exactly the ones every consumer reads. A
mismatch there does not crash: each script hits its `if not exists: continue` guard and the
run finishes with no output and no error. It also checks cross-script imports resolve, that
scripts sharing the weight-edit helpers agree on which projections they touch, and that
every flag and script named in the docs actually exists.

Run it after any edit, rename or port. Exit code is non-zero on failure, so it drops
straight into CI or a pre-commit hook.

---

## Provenance

The directional-ablation operator is from:

> Arditi A, Obeso O, Syed A, Paleka D, Panickssery N, Gurnee W, Nanda N.
> **Refusal in language models is mediated by a single direction.**
> arXiv:[2406.11717](https://arxiv.org/abs/2406.11717) (2024).

That paper's finding — that a behaviour as complex as refusal is approximately
one-dimensional in activation space — is what makes this class of intervention work at all.

This implementation was written for nanoGPT **after reading
[Heretic](https://github.com/p-e-w/heretic)** (Philipp Emanuel Weidmann, AGPL-3.0), a
reference implementation of directional ablation for large language models. Two things were
taken from it: the **layer-indexing convention** (layer 0 = embedding output, layers 1..n =
block outputs) and the **choice to target both residual-writing projections**.

**No Heretic code is copied or imported.** These scripts depend only on numpy, torch,
scikit-learn, scipy, matplotlib and pacmap, plus your model's own definition. The
implementation is therefore not a derivative work and the AGPL does not propagate — but
credit is due and is recorded here.

Projection method:

> Wang Y, Huang H, Rudin C, Shaposhnik Y. **Understanding how dimension reduction tools
> work.** *JMLR* 22(201):1–73 (2021). [PaCMAP](https://github.com/YingfanWang/PaCMAP)

---

## Status

Working research code, not a packaged library. Written for one study and generalised
afterwards; expect to edit paths and module names.

Results obtained with these tools are reported separately (manuscript in preparation) and
are not included here.

## Licence and citation

**[PolyForm Noncommercial License 1.0.0](LICENSE.md)**, with Additional Terms.

You may **use, modify and redistribute** this code, including for academic research, on
these conditions:

- **Noncommercial only.** Commercial use requires a separate licence from the copyright
  holder. PolyForm defines permitted noncommercial purposes precisely — see
  *Noncommercial Purposes*, *Personal Uses* and *Noncommercial Organizations* in
  [`LICENSE.md`](LICENSE.md).
- **Citation is mandatory, not a courtesy.** Additional Term 1 makes it a condition of the
  licence: any use of the software or any part or derivative of it — in research, software,
  products, publications, presentations, or models built with its help — must credit the
  project and cite this repository. Academic works must cite using
  [`CITATION.cff`](CITATION.cff).
- **Licence propagation.** Every copy, fork or derivative must carry the whole
  `LICENSE.md` unaltered, including the `Required Notice:` lines (Additional Term 2).

Provided **as-is, with no warranty or condition of any kind**, and the licensor accepts
**no liability** for any damages arising from the software or its use or misuse
(*No Liability*, `LICENSE.md`).

### How to cite

> Salehi-Ashtiani, K. *Res-Lens: a lens on the residual stream — and a scalpel for it.*
> https://github.com/kouroshSA/Res-Lens

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders a
"Cite this repository" button from it.

If you use the ablation or steering operator, please also cite its origin — Arditi et al.,
*Refusal in language models is mediated by a single direction*, arXiv:2406.11717 (2024).

> **Note.** PolyForm Noncommercial is a software-native licence, better drafted for code
> than a Creative Commons licence, but it is **not** OSI-approved open source. Some
> institutions and downstream projects decline non-commercial dependencies; that is a real
> cost to adoption, accepted deliberately here. PolyForm alone does not require citation —
> the Additional Terms in `LICENSE.md` add that obligation.

