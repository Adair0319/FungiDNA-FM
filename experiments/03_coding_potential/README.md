# Experiment 3 — Coding potential prediction

**Produces:** Figure 4, Supplementary Table S3 (Methods 2.8)

Binary classification of CDS versus intergenic sequence, plus two small-ORF
transfer settings.

## Main dataset (Fig 4A)

From ~9.8M CDS and ~9.6M intergenic sequences across the 734 genomes.
Positives are deduplicated with **CD-HIT at 90% sequence similarity**, filtered
to 300–5,000 bp, split into 20 equal-frequency length bins, 5,000 per bin →
100,000 CDS. Negatives are length-matched the same way → 100,000 intergenic.
Balanced 200,000-sample dataset, 5-fold stratified by class label and species.

```bash
python build_cds_intergenic_dataset.py
python train_cds_intergenic.py --model ours --mode finetune   # ours|gena|cnn|mamba2
```

Six reported combinations: FungiDNA (frozen / fine-tuned), GENA-LM (frozen /
fine-tuned), CNN and Mamba2-1layer (end-to-end).

## Few-shot (Fig 4B)

Class-balanced nested subsets at 10 / 20 / 30 / 40% of the training labels,
with 100% as reference. Fully fine-tuned FungiDNA vs GENA-LM, 5 folds.

```bash
python prepare_fewshot_data.py && python prepare_fewshot_folds.py
bash run_fewshot_all_folds.sh
```

## Fungal sORFs (Fig 4C)

Annotated fungal CDS ≤ 300 bp as positives, length-matched random intergenic
crops as negatives.

```bash
python prepare_hard_cds_data.py
```

## Human sORFs (Fig 4D)

External transfer test on a **CPPred**-derived set: 641 short coding RNAs and
641 length-matched non-coding RNAs (ORFs below ~303 nt).

```bash
python prepare_sorf_data.py
```

Reported caveat: FungiDNA fine-tuning is unstable on this small external set —
3 of 5 folds collapsed, MCC SD ≈ 0.26.

## Metrics and results

F1, MCC, AUROC, AUPRC. Collected under
[`results/coding_potential/`](../../results/coding_potential/)
(`main/`, `fewshot/`, `fungal_sorf/`, `human_sorf/`).

## External tool

CD-HIT (Fu et al. 2012) is required for dataset construction and is not
vendored here — install it separately.
