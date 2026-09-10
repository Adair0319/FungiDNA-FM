# FungiDNA-FM

A genomic foundation model pretrained on diverse fungal whole genomes.

FungiDNA-FM (113 M parameters) is a hybrid **Mamba2 + self-attention** backbone
trained on 39.48 Gb from 734 fungal genomes spanning six phyla. It produces
representations that transfer to five downstream tasks: representation
clustering, taxonomic classification, coding-potential prediction, splice-site
classification, and biosynthetic-gene-cluster (BGC) boundary regression.


## Model at a glance

|                    |                                                              |
| ------------------ | ------------------------------------------------------------ |
| Backbone           | **StripedMamba** — 24 layers as six blocks of (3 × Mamba2 + 1 × self-attention) |
| Parameters         | ~113 M, hidden dim 768                                       |
| Mamba2 layers      | state dim 128, conv width 4, expansion 2 (inner dim 1,536)   |
| Attention layers   | 12 heads × 64 dims, FlashAttention-2, SwiGLU (intermediate 2,048) |
| Normalisation      | RMSNorm throughout, dropout 0, RoPE (base 10,000, up to 131,072 positions) |
| Tokenizer          | SentencePiece **Unigram**, vocab 4,096, ≈ 3 bp per token     |
| Pooling            | mean over non-padding positions (not `[CLS]`)                |
| Pretraining corpus | 734 fungal genomes, **39.48 Gb**, six phyla (JGI MycoCosm *1000 Fungal Genomes*) |
| Pretraining        | Phase 1 MLM (1,024 / 2,048 / 4,096 bp) → Phase 2 MLM + supervised contrastive (8,192 bp) |

## Model weights

The pretrained backbone and downstream-task checkpoints are hosted on the
Hugging Face Hub:

**[https://huggingface.co/Biopotato/FungiDNA-FM](https://huggingface.co/Biopotato/FungiDNA-FM)**

The repository contains:

```
Adair0319/FungiDNA-FM
├── backbone_final.pt                    # pretrained backbone
├── tokenizer/{bpe_fungi.model, .vocab}  # SentencePiece tokenizer
├── coding_potential/best_model.pt
├── splice_site/best_model.pt
├── taxonomy/{phylum,subphylum,class,order,family}/mlp_best.pt
└── bgc_boundary/best_model.pt
```

(If your Hugging Face username differs, adjust the URL accordingly. See
[`docs/weights_manifest.md`](docs/weights_manifest.md) for the exact files and
their sizes.)

Download the files (e.g. via `git lfs` / `huggingface-cli download` / the Hub
web UI) to a local directory, then tell the handlers where they live with a
`~/.config/fungidna/weights.yaml` config:

```bash
mkdir -p ~/.config/fungidna
cp configs/weights.example.yaml ~/.config/fungidna/weights.yaml
# edit the paths to point at your downloaded files
```

```yaml
# ~/.config/fungidna/weights.yaml
backbone: /path/to/backbone_final.pt
tokenizer: /path/to/tokenizer/bpe_fungi.model
coding_potential: /path/to/coding_potential/best_model.pt
splice_site: /path/to/splice_site/best_model.pt
taxonomy: /path/to/taxonomy/phylum/mlp_best.pt
bgc_boundary: /path/to/bgc_boundary/best_model.pt
```

The same paths can be given via environment variables
(`FUNGIDNA_BACKBONE_WEIGHTS`, `FUNGIDNA_TOKENIZER_WEIGHTS`, and
`FUNGIDNA_<TASK>_WEIGHTS`).

## Installation

```bash
conda env create -f environment.yml
conda activate fungidna
```

`flash-attn` and `mamba-ssm` need a CUDA toolchain matching your PyTorch build;
if the conda solve is slow, install the pinned pip wheels from
`requirements.txt` into a bare Python 3.10 + PyTorch 2.5.1 + CUDA 12.1 env
instead.

## Quick start

FungiDNA-FM exposes two equivalent entry points that share one validation gate.

**Embedded API:**

```python
from fungidna.invoke import run_task

report = run_task(file_path="/path/test.fasta", task="coding_potential")
print(report.to_json())
```

**Prompt-driven terminal:**

```bash
python -m fungidna "Please predict the splice sites using /path/test.fasta"
```

Requests are validated before execution: unsupported tasks, missing files, and
malformed FASTA/FASTQ inputs return a structured error and never reach the
model.

**Demo:**
The following demo shows the validation gate, prompt-driven terminal entry point,
task dispatch, and downstream prediction output.

▶️ Watch the full demo(demo.mp4)

## Supported tasks

| Task                     | Canonical name     | Description                                           |
| ------------------------ | ------------------ | ----------------------------------------------------- |
| Representation           | `representation`   | 768-d mean-pooled embeddings per sequence             |
| Taxonomic classification | `taxonomy`         | phylum label per sequence (five-rank heads available) |
| Coding potential         | `coding_potential` | coding vs non-coding probability per sequence         |
| Splice site              | `splice_site`      | donor / acceptor / non-site per candidate motif       |
| BGC boundary             | `bgc_boundary`     | left/right boundary distance in bp                    |

Aliases are accepted (e.g. `"splice site"`, `"taxonomic classification"`,
`"embeddings"`) and normalised to the canonical names above.

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

## Repository layout

```
fungidna/          core library — model, data, training, and the invoke workflow
data_prep/         raw JGI data → per-task datasets
experiments/       one directory per paper experiment
baselines/         baseline model definitions and download helper
configs/           pretraining configs and the weights config template
figures/           figure and report generators
docs/              genome list, weights manifest, and supporting documentation
```

## Data

The genome assemblies and annotations come from the JGI MycoCosm *1000 Fungal
Genomes* initiative and are **not** redistributed here. The exact set used is
listed in [`docs/genome_list.tsv`](docs/genome_list.tsv) — 735 portal IDs with
their phylum, subphylum, class, order and family. One assembly is dropped by
QC, leaving the 734 genomes reported in the paper.

Download the assemblies and GFF3 annotations from MycoCosm by portal ID, then
run the scripts in [`data_prep/`](data_prep/). CD-HIT (Fu et al. 2012) is
required for the coding-potential dataset and is not vendored here.


## License

[MIT](LICENSE). Note that the pretrained baselines downloaded by
`baselines/download.py` carry their own licenses — GENA-LM is CC-BY-NC-SA 4.0.

## Citation

Citation details will be added once the paper is published.
