# Social-Bias-in-Stable-Diffusion: Tracing Social Bias Through the Cross-Attention Lens

Auditing where occupational bias lives in Stable Diffusion 1.5 i.e across CLIP, UNet cross-attention, and LCM LoRA

**Where does Stable Diffusion's bias live?**

An audit of occupational bias in Stable Diffusion 1.5 across three pipeline
stages — CLIP prompt embeddings, UNet cross-attention, and denoising time —
for 37 occupations spanning the full BLS gender-imbalance spectrum.

Narmeen Sabah Siddiqui · Deep Learning for Computer Vision II · May 2026

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/narmeensabahsiddiqui/social-bias-in-sd)
[![Dataset on Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/narmeensabahsiddiqui/professions-sd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Headline findings

- **The CLIP text encoder is the primary site of the bias.** A Bolukbasi-style
  directional gender measure on CLIP's pooled output predicts SD's per-profession
  output gender skew at **Pearson r = +0.896** (p ≈ 10⁻¹⁴, n = 37).
- **Bias is not always amplification.** SD generates *professor* as 10% women
  against a BLS reality of 52.3% — a 42-point inversion. Combined with the race
  analysis, SD's professor is also 55% East Asian: three demographic axes
  diverge from BLS reality simultaneously.
- **Bias is contextual, not just demographic.** Civil engineer's profession-token
  attention is spatially indistinguishable from construction worker's (both
  attend to construction sites and hard hats) — a workplace conflation
  invisible at the demographic level.
- **Gender locks in early.** Per-timestep attention trajectories stabilise
  by step 5–8 of 30 and don't separate by gender afterward.
- **LCM LoRA preserves bias at the extremes and amplifies it on borderline
  professions.** Four-step inference saves 5.2× compute but costs 10–26 pp
  on exactly the professions where SD baseline was producing counter-
  stereotypical diversity.
- **The audit is cheap.** Full pipeline cost 89.6 g CO₂ (198 Wh) — about
  0.5 km of car driving.


## Reproducing the results

The notebook runs end-to-end in two modes:

- **From cache (~7 min on a Kaggle T4)**: attach the published dataset
  [`professions-sd`](https://www.kaggle.com/datasets/narmeensabahsiddiqui/professions-sd)
  as `/kaggle/input/professions-sd` and run all. Expensive generation cells
  skip automatically and read from cache.
- **From scratch (~2 hr on a Kaggle T4)**: run without the dataset attached.
  All generation cells execute.

```bash
# Local reproduction (requires CUDA GPU with ≥ 16GB VRAM)
pip install -r requirements.txt
jupyter notebook notebooks/social-bias-in-sd.ipynb
```

## Methodology at a glance

| RQ | Question | Method | Main finding |
|---|---|---|---|
| RQ1 | Does SD skew gender vs BLS? | 37 professions × 20 images, MTCNN + FairFace ViT | Systematic skew; sometimes *inversion* not amplification |
| RQ2 | Is bias already in CLIP? | Bolukbasi-style gender direction, directional projection | r = +0.896 — CLIP ships ~80% of the bias |
| RQ3 | Where in the UNet? | Custom AttnProcessor, cross-attention attribution, object-leakage probe | Workplace conflation (civil engineer ↔ construction site) |
| RQ4 | When during denoising? | Per-timestep attention from the same forward pass | Gender locks in by step 5–8 of 30 |
| RQ5 | What's the carbon cost? | CodeCarbon tracking | 89.6 g CO₂ for the full audit |
| RQ6 | Does LCM LoRA preserve bias? | Re-audit at 4 steps vs 30 | Preserves extremes, amplifies borderlines |
| Race | Intersectional analysis | FairFace race ViT, Bonferroni-corrected sweep | 5 professions survive correction, multi-axis prototypes |

## Methodological contributions

1. **A correctly-implemented Bolukbasi-style directional bias estimator for
   CLIP sentence embeddings**, with a documented mid-project bug-and-fix.
2. **A custom `AttnProcessor` for `diffusers` 0.36** that records per-timestep
   cross-attention attribution in a single forward pass — re-implementing
   DAAM's technique for the current diffusers API.
3. **A descriptive object-leakage probe** via CLIP zero-shot classification of
   high-attention image crops that surfaces workplace-conflation bias.
4. **A Bonferroni-corrected intersectional race analysis** on the saved
   images, demonstrating that SD encodes coupled multi-axis prototypes.

## Data

- **Prompts**: 37 occupations × 20 images = 740 generations for RQ1
- **Attention traces**: 7 professions × 30 images = 210 for RQ3/RQ4
- **LCM comparison**: 7 professions × 10 images = 70 for RQ6
- **Baselines**: BLS CPS 2025 Annual Averages, Table 11

Full generated dataset (images, attention maps, CSVs) is published at
[kaggle.com/datasets/narmeensabahsiddiqui/professions-sd](https://www.kaggle.com/datasets/narmeensabahsiddiqui/professions-sd).

## Status

Course project submitted May 2026 for Deep Learning for Computer Vision II
(IPCVai Erasmus Mundus programme). Project supervisor: Marcos Escudero-Viñolo.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{siddiqui2026sdbias,
  author = {Siddiqui, Narmeen Sabah},
  title  = {Tracing Social Bias Through the Cross-Attention Lens:
            Where Does Stable Diffusion's Bias Live?},
  year   = {2026},
  note   = {IPCVai Individual Project},
  howpublished = {\url{https://github.com/NARMYN/social-bias-in-sd}}
}
```

