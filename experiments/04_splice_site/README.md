# Experiment 4 — Splice-site classification

**Produces:** Figure 5, Supplementary Table S4 (Methods 2.9)

Three-class classification — donor / acceptor / non-site — from 401-bp windows
(200 bp either side of the candidate position), derived from the GFF3
annotations of the 734 genomes.

## Dataset construction

Balanced 1:1:1, capped at 200,000 samples. Filters applied:

- retain common dinucleotides — donors GT, GG, TG, AG, CG; acceptors AG, GG,
  GA, GC, GT
- remove ambiguous bases, within-class duplicates and cross-label conflicts
- intron length 50–500 bp; transcripts with ≥ 3 exons
- per-species caps and stratified downsampling

Splitting is **GroupKFold** on a composite transcript identifier, so every
sample from a transcript lands in the same fold; within the training pool,
90/10 train/val.

```bash
python -m fungidna.data.splice_site_dataset_v2      # build per-species parquets
python -m fungidna.data.splice_site_combine_v2      # GroupKFold split + stats
python preprocess_splice_site.py                    # tokenise to memmap
python precompute_dna_memmap.py                     # 401-bp uint8 memmap
```

## Experiments

| Script | Role |
|---|---|
| `run_splice_v2_experiments.py` | FungiDNA frozen and fully fine-tuned, plus the CNN baseline; 5 experiment types × 5 folds |
| `run_gena_splice.py` | GENA-LM baseline (frozen and fine-tuned) |
| `run_splice_v2_simple_baselines.py` | single-layer Mamba2 baseline |
| `aggregate_splice_v2_results.py` | aggregate the 5-fold results |
| `launch_splice_v2_experiments.sh` | launcher |

Few-shot at 10 / 20 / 30 / 40% of training labels (100% as reference), fully
fine-tuned FungiDNA vs GENA-LM.

## Metrics and results

F1, MCC, AUROC, AUPRC overall, plus class-wise AUROC / AUPRC / F1 / recall
(Fig 5B focuses on acceptor sites).

Collected under [`results/splice_site/`](../../results/splice_site/) —
`fullft_lr2e-5`, `frozen`, `cnn`, `baseline_gena`, `baseline_gena_frozen`,
`baseline_mamba2_1layer`, `fewshot`.
