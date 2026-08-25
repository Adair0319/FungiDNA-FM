# FungiDNA-FM

## Model at a glance

| | |
|---|---|
| Backbone | **StripedMamba** — 24 layers as six blocks of (3 × Mamba2 + 1 × self-attention) |
| Parameters | ~113 M, hidden dim 768 |
| Mamba2 layers | state dim 128, conv width 4, expansion 2 (inner dim 1,536) |
| Attention layers | 12 heads × 64 dims, FlashAttention-2, SwiGLU (intermediate 2,048) |
| Normalisation | RMSNorm throughout, dropout 0, RoPE (base 10,000, up to 131,072 positions) |
| Tokenizer | SentencePiece **Unigram**, vocab 4,096, ≈ 3 bp per token |
| Pooling | mean over non-padding positions (not `[CLS]`) |
| Pretraining corpus | 734 fungal genomes, **39.48 Gb**, six phyla (JGI MycoCosm *1000 Fungal Genomes*) |
| Pretraining | Phase 1 MLM (1,024 / 2,048 / 4,096 bp) → Phase 2 MLM + supervised contrastive (8,192 bp) |
| Compute | 8 × NVIDIA L20 (46 GB), Python 3.10, PyTorch 2.5.1, CUDA 12.1 |


## Paper → code

| Paper                                     | Path                                                         | What is there                                                |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Methods 2.1 — genome data & preprocessing | [`data_prep/genomes/`](data_prep/genomes/)                   | JGI file organisation, genomic-region extraction, QC (contigs ≥ 5 kb, ≤ 10% N) |
| Methods 2.2 — DNA tokenizer               | [`fungidna/data/tokenizer.py`](fungidna/data/tokenizer.py)   | SentencePiece Unigram, vocab 4,096                           |
| Methods 2.3 — model architecture          | [`fungidna/model/`](fungidna/model/)                         | `striped_mamba.py`, `mamba2_block.py`, `attention_block.py`, `embedding.py`, `heads.py` |
| Methods 2.4 — two-phase pretraining       | [`fungidna/training/`](fungidna/training/), [`experiments/00_pretraining/`](experiments/00_pretraining/) | `phase1_mlm.py`, `phase2_joint.py`, `losses.py`, launchers, configs |
| Methods 2.5 — baselines                   | [`baselines/`](baselines/)                                   | CNN and Transformer definitions; GENA-LM download            |
| Methods 2.6 · **Fig 2** · Table S1        | [`experiments/01_representation_clustering/`](experiments/01_representation_clustering/) | UMAP + K-means over phylum and genomic-element representations |
| Methods 2.7 · **Fig 3** · Table S2        | [`experiments/02_taxonomic_classification/`](experiments/02_taxonomic_classification/) | Five-rank classification, five-fold CV and genome-isolated   |
| Methods 2.8 · **Fig 4** · Table S3        | [`experiments/03_coding_potential/`](experiments/03_coding_potential/) | CDS vs intergenic, few-shot, fungal sORFs, human sORFs (CPPred) |
| Methods 2.9 · **Fig 5** · Table S4        | [`experiments/04_splice_site/`](experiments/04_splice_site/) | Donor / acceptor / non-site, 401-bp windows, GroupKFold      |
| Methods 2.10 — environment                | [`environment.yml`](environment.yml), [`requirements.txt`](requirements.txt) |                                                              |
| Figures 2–5                               | [`figures/`](figures/)                                       | Report and figure generators                                 |

Each `experiments/NN_*/` directory has its own README naming the figure and
table it produces, the dataset it needs, and the commands to run.

## Repository layout

```
fungidna/          core library — model, data, training
data_prep/         raw JGI data → per-task datasets
experiments/       one directory per paper experiment
baselines/         baseline model definitions and download helper
configs/           pretraining configs
figures/           figure and report generators
docs/              genome list and supporting documentation
metadata/          taxonomy table consumed by fungidna/data/taxonomy.py
```

## Installation

```bash
conda env create -f environment.yml
conda activate fungidna
```

`flash-attn` and `mamba-ssm` need a CUDA toolchain matching your PyTorch build;
if the conda solve is slow, install the pinned pip wheels from
`requirements.txt` into a bare Python 3.10 + PyTorch 2.5.1 + CUDA 12.1 env
instead.

## Data

The genome assemblies and annotations come from the JGI MycoCosm *1000 Fungal
Genomes* initiative and are **not** redistributed here. The exact set used is
listed in [`docs/genome_list.tsv`](docs/genome_list.tsv) — 735 portal IDs with
their phylum, subphylum, class, order and family. One assembly is dropped by
QC, leaving the 734 genomes reported in the paper.

Download the assemblies and GFF3 annotations from MycoCosm by portal ID, then
run the scripts in [`data_prep/`](data_prep/).

CD-HIT (Fu et al. 2012) is required for the coding-potential dataset and is not
vendored here.

## License

[MIT](LICENSE). Note that the pretrained baselines downloaded by
`baselines/download.py` carry their own licenses — GENA-LM is CC-BY-NC-SA 4.0.

## Citation

Citation details will be added once the paper is published.
