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

## What "ablating a direction" means

Easy to misread, so plainly:

**It is not a token, a word, or an embedding.** Nothing is removed from the vocabulary and
**the embedding table is never modified.**

**It is a direction in activation space.** Average the residual stream over many class-A
inputs, average over many class-B inputs, subtract:

```
v = normalize( mean(class A) − mean(class B) )
```

Individual inputs average out; what survives is the axis along which the model separates the
two classes.

**The edit removes the model's ability to write along that axis.** Each block writes into the
residual stream through projection matrices. Replacing `W` with `(I − v vᵀ)W` leaves them
able to compute everything else, but their output has exactly zero component along `v`.
Later layers never see the feature.

```
W  ←  W − λ · v (vᵀ W)          λ = 1  ⇒  (I − v vᵀ) W
```

No retraining, no gradients — one rank-1 edit per matrix.

Because a *learned feature computed from the whole input* is removed rather than a lexical
item, the intervention generalises to inputs never used to compute `v` — which `splithalf.py`
tests directly. And because one direction out of `d_model` is removed, collateral damage is
small.

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
| `ablate.py` | the causal ablation, with random-direction and PC1 controls and a λ sweep |
| `splithalf.py` | estimates `v` on one half, ablates, evaluates on the other — the circularity control |
| `aggregate_geometry.py` | cross-model layer curves and per-model tables |
| `aggregate_ablation.py` | cross-model ablation summary and specificity gaps |
| `pooled_projection.py` | pools replicates into one projection per layer; reports whether it separates by **class** or by **model identity** |
| `figures_combined.py` | combined class-only figures, light and dark |
| `error_margin.py` | is model error predictable from geometry, beyond the decision variable itself? |
| `collateral.py` | does the edit break the **decision** or the **model**? task damage vs general next-token KL |
| `steer.py` | adds `±αv` instead of removing it — **sufficiency**, with a saturation check |

Each writes figures (300 dpi PNG + vector PDF), a `metrics.json`, and where relevant a
`.csv`. Every plotted number is recomputed from the inputs.

**See [`USAGE.md`](USAGE.md)** for run order, arguments, sanity checks and a pitfalls
section.

---

## Adapting to your model

Class names in the code are **PRS** and **RRS** (positive / random reference set — standard
protein-interaction vocabulary). They are just labels for "class A" and "class B"; the method
is agnostic to what they mean.

To port: change the module names in `ablate.py::ablate_`, the hook targets in
`residual_extract.py::ResidualCatcher`, and the readout in the scoring functions.

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

**[CC BY-NC 4.0](LICENSE)** — Creative Commons Attribution-NonCommercial 4.0 International.

You may **use, modify and redistribute** this code, including for academic research, on two
conditions:

- **Non-commercial only.** Use primarily intended for or directed toward commercial
  advantage or monetary compensation is not permitted under this licence. For commercial
  use, contact the author.
- **Attribution.** If you use or modify any part of this repository, you must give
  appropriate credit, link to the licence, and indicate whether changes were made.

Provided **as-is, without warranties of any kind**, and the author accepts **no liability**
for any use or misuse (LICENSE §5).

### How to cite

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders a
"Cite this repository" button from it. In text:

> Salehi Ashtiani, K. *Res-Lens: a lens on the residual stream — and a scalpel for it.*
> https://github.com/kouroshSA/Res-Lens (CC BY-NC 4.0).

If you use the ablation or steering operator, please also cite its origin — Arditi et al.,
*Refusal in language models is mediated by a single direction*, arXiv:2406.11717 (2024).

> **Two notes, offered as information rather than legal advice.** (1) A non-commercial
> licence is **not** OSI-approved open source; some institutions and downstream projects
> decline such dependencies, which is a real cost to adoption if you later want uptake.
> (2) Creative Commons does not itself recommend CC licences for software, since they do
> not address source/object distinction or patent grants. It was chosen here because it
> matches the stated requirements — non-commercial, attribution mandatory, no warranty, no
> liability — more directly than any software licence does.
> [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
> is the software-native alternative, but it does **not** legally require citation; it
> would make attribution a request rather than a condition. Swapping is a one-file change.
