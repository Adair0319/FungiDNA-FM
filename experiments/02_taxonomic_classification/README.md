# Experiment 2 — Fungal taxonomic classification

**Produces:** Figure 3, Supplementary Table S2 (Methods 2.7)

Classification at five ranks (phylum, subphylum, class, order, family) from
non-overlapping 10-kb genomic windows. Windows with > 5% ambiguous bases are
removed and each genome contributes at most 600 windows. Genomes labelled
*Incertae sedis* are excluded per rank (24 at subphylum, 87 at class, 17 at
order, 49 at family).

Two evaluation settings:

- **Five-fold CV** — windows shuffled within genome; each fold tested once, the
  remaining folds split 90:10 train/val. Splits windows, not genomes.
- **Genome-isolated** — whole genomes assigned to train or test.

Note: genome isolation applies to downstream classifier fitting only. It does
**not** exclude the test genomes or species from FungiDNA pretraining.

## Dataset construction

```bash
python build_five_rank_cv_datasets.py          # five-fold CV sets
python build_class_order_family_datasets.py    # genome-isolated, class/order/family
python build_phylum_subphylum_datasets.py      # genome-isolated, phylum/subphylum
```

## Evaluation

| Script | Role |
|---|---|
| `train_phase2_joint_mlp.py` | FungiDNA (mean-pooled 768-dim → MLP), five-fold CV |
| `eval_genome_isolated.py` | all models, genome-isolated, all five ranks |
| `evaluate_baselines.py` | GENA-LM (frozen), CNN and Transformer (end-to-end) |
| `run_five_rank_mamba2_baseline.py` | single-layer Mamba2 baseline, five-fold CV |
| `run_genome_isolated_mamba2.py` | single-layer Mamba2 baseline, genome-isolated |
| `recompute_mamba2_auroc.py` | robust macro AUROC/AUPRC for the Mamba2 baselines |

Metrics: macro F1, MCC, macro AUROC, macro AUPRC.

Results: [`results/five_rank/`](../../results/five_rank/) and
[`results/genome_isolated/`](../../results/genome_isolated/), each with
`{phylum,subphylum,class,order,family}/{ours,gena,cnn,mamba2}.json`.

## A note on the shipped scripts

`evaluate_baselines.py` and `eval_genome_isolated.py` retain DNABERT-2 / NT-v2
code paths from exploratory work. Those models are **not** paper baselines and
have been removed from the `--model` choices.
