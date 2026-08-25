# Pretraining

Two-phase pretraining of FungiDNA (Methods 2.4). Produces the backbone used by
every downstream experiment.

## Corpus

734 fungal whole genomes (39.48 Gb) from the JGI MycoCosm *1000 Fungal Genomes*
initiative, spanning six phyla. Genome list: [`docs/genome_list.tsv`](../../docs/genome_list.tsv).
QC (`fungidna/data/qc_filter.py`): contigs ≥ 5,000 bp and ≤ 10% ambiguous bases,
from an initial 735 assemblies.

## Phase 1 — masked sequence learning

Random segments of 1,024 / 2,048 / 4,096 bp, adjacent windows overlapping 75%.
15% of tokens selected for prediction; 30% of those grouped into contiguous
spans (geometric, λ = 3.0), 70% independent. MLM output weights tied to the
input embedding matrix.

```bash
bash train_phase1.sh          # 8 GPUs; config: configs/pretrain_phase1.yaml
```

AdamW, peak LR 5e-4, weight decay 0.1, grad-norm clip 1.0, linear warmup
10,000 steps then cosine decay. Effective batch = 8 GPUs × 32 seqs × 8 accum
= 2,048 sequences.

## Phase 2 — MLM combined with contrastive learning

Initialised from the Phase-1 checkpoint. Non-overlapping 8,192-bp windows from
contigs ≥ 50 kb, windows with > 5% ambiguous bases removed (~4.3M sequences).
Batch = 8 genomes × 4 windows. One backbone pass feeds both the MLM head and
the contrastive projection head; total loss is a weighted sum.

```bash
bash train_phase2_joint.sh    # config: configs/pretrain_phase2_joint.yaml
```

Layer-wise LRs: 1e-4 backbone (incl. LM head), 5e-4 projection head; both
warmed up 5,000 steps then cosine decay.

Note: `use_weighted_cl` is `false` in the shipped config — the paper uses the
standard supervised contrastive loss.

## Compute

8 × NVIDIA L20 (46 GB) on an Inspur NF5280M6. bfloat16 AMP, gradient
checkpointing on all 24 layers.
