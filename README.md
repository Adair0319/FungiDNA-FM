# FungiDNA-FM

A genomic foundation model pretrained on diverse fungal whole genomes.

FungiDNA-FM (113 M parameters) is a hybrid **Mamba2 + self-attention** backbone
trained on 39.48 Gb from 734 fungal genomes spanning six phyla. It produces
embeddings that transfer to five downstream tasks: taxonomy
classification, CDS/intergenic discrimination, short-CDS discrimination,
splice-site classification, and biosynthetic-gene-cluster (BGC) boundary
regression — plus an embedding extraction capability.

Companion repository for the paper *"FungiDNA: a genomic foundation model
pretrained on diverse fungal whole genomes."*

## Variants

FungiDNA-FM is released as two variants that share the same StripedMamba
backbone:

| Variant | Tokenizer |
|---|---|
| **FungiDNA-FM-SNT** | single-nucleotide (A/C/G/T/N) |
| **FungiDNA-FM-Unigram** | SentencePiece Unigram, vocab 4,096 |

## Model at a glance

| | |
|---|---|
| Backbone | **StripedMamba** — 24 layers as six blocks of (3 × Mamba2 + 1 × self-attention) |
| Parameters | ~113 M, hidden dim 768 |
| Mamba2 layers | state dim 128, conv width 4, expansion 2 (inner dim 1,536) |
| Attention layers | 12 heads × 64 dims, FlashAttention-2, SwiGLU (intermediate 2,048) |
| Normalisation | RMSNorm throughout, dropout 0 |
| Tokenizer | SentencePiece **Unigram**, vocab 4,096, ≈ **5.6 bp/token** |
| Mask token | SentencePiece Unigram mask id = **2** |
| Pooling | mean over non-padding positions (not `[CLS]`) |
| Pretraining corpus | 734 fungal genomes, **39.48 Gb**, six phyla (JGI MycoCosm *1000 Fungal Genomes*) |
| Pretraining | Phase 1 MLM (1,024 / 2,048 / 4,096 bp) → Phase 2 MLM + supervised contrastive (8,192 bp) |

## Model weights

Weights are hosted at <https://huggingface.co/Biopotato/FungiDNA-FM>. The two
variants' final Phase-2 checkpoints are:

| Variant | Backbone | Checkpoint |
|---|---|---|
| FungiDNA-FM-SNT | `snt/backbone.pt` | final Phase 2 checkpoint, step 33,000 |
| FungiDNA-FM-Unigram | `unigram/backbone.pt` | final Phase 2 checkpoint, step 16,000 |

Download the files (e.g. via `huggingface-cli download` or the Hub web UI) to a
local directory, then point a `~/.config/fungidna/weights.yaml` at them:

```bash
mkdir -p ~/.config/fungidna
cp configs/weights.example.yaml ~/.config/fungidna/weights.yaml
# edit the paths to point at your downloaded files
```

Each variant has its own section, so the loader can never mix their backbones:

```yaml
tokenizer: /path/to/tokenizer/bpe_fungi.model

unigram:
  backbone: /path/to/unigram/backbone.pt
  cds: /path/to/unigram/tasks/cds/best_model.pt

snt:
  backbone: /path/to/snt/backbone.pt
  cds: /path/to/snt/tasks/cds/best_model.pt
```

Environment overrides are variant-scoped, e.g.
`FUNGIDNA_UNIGRAM_BACKBONE_WEIGHTS` and `FUNGIDNA_SNT_BACKBONE_WEIGHTS`.

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

**Representation — 768-d embedding per sequence:**

```python
from fungidna.representation import RepresentationModel

model = RepresentationModel(variant="unigram")   # or variant="snt"
vector = model.embed("ATGGCTAGCTAGCTA")          # (768,)
matrix = model.embed(["ACGTACGT...", "TTTT..."]) # (n, 768)
```

Each variant is mean-pooled over its DNA tokens only: no `[CLS]` is added and
padding is excluded, matching the representation used in the paper. The model
can also be driven from a FASTA file with `run_task(..., task="embedding")`.

**Task prediction — two equivalent entry points that share one validation gate.**

**Embedded API:**

```python
from fungidna.invoke import run_task

report = run_task(file_path="/path/test.fasta", task="cds", variant="unigram")
print(report.to_json())

# taxonomy: pick a rank and read both the predicted id and the category name
tax = run_task(file_path="/path/genome.fasta", task="taxonomy",
               variant="unigram", rank="subphylum")
tax.results["labels"]        # {header: predicted class id}
tax.results["labels_named"]  # {header: predicted category name}
```

Set `variant="snt"` for the single-nucleotide variant. The `variant` selects the
tokenizer, the `token_type` (0 for Unigram, 1 for SNT), and the matching backbone.

**Prompt-driven terminal:**

```bash
python -m fungidna "Please predict the splice sites using /path/test.fasta"
```

Requests are validated before execution: unsupported tasks, missing files, and
malformed FASTA/FASTQ inputs return a structured error and never reach the
model.

## Supported tasks

| Task | Canonical name | Description |
|---|---|---|
| Embedding | `embedding` | 768-d mean-pooled embeddings per sequence |
| Taxonomic classification | `taxonomy` | phylum-to-family classification per 10 kb window |
| CDS / intergenic | `cds` | CDS vs intergenic discrimination (300–5,000 bp) |
| Short CDS | `short_cds` | short-CDS vs intergenic-ORF discrimination (≤ 300 bp) |
| Splice site | `splice_site` | donor / acceptor / non-site per 401 bp window |
| BGC boundary | `bgc_boundary` | left/right boundary distance in bp (10 kb window) |

Aliases are accepted and normalised to the canonical names above.

> Note: the released `short_cds` and `splice_site` checkpoints are the
> best-validation checkpoints of **fold 4** (the last of the five folds), not a
> fold selected as the best-performing across the five-fold benchmark. The
> paper's reported five-fold mean ± std is a separate result and is not the
> same as the single released checkpoint's performance.

## Paper → code

The paper uses Supplementary Notes S1–S6; the table maps them to this
repository.

| Note | Path | What is there |
|---|---|---|
| S1 — corpus | [`data_prep/`](data_prep/), [`docs/genome_list.tsv`](docs/genome_list.tsv) | genome acquisition, QC, the exact genome list |
| S2 — architecture & pretraining | [`fungidna/model/`](fungidna/model/), [`fungidna/training/`](fungidna/training/), [`scripts/`](scripts/), [`configs/`](configs/) | StripedMamba model, Phase 1 / Phase 2 trainers, launchers, configs |
| S3 — downstream tasks | [`fungidna/invoke/`](fungidna/invoke/) | task handlers and the guarded invocation interface |
| S4–S6 — evaluation, input/optimisation, masked recovery | [`supplementary/`](supplementary/) | supplementary tables and configuration notes (to be added) |

## Repository layout

```
fungidna/          core library — model, data, training, and the invoke workflow
data_prep/         raw JGI data → pretraining corpus
scripts/           pretraining launchers (train_phase1.sh, train_phase2_joint.sh)
configs/           pretraining configs and the weights config template
docs/              genome list, weights manifest, and supporting documentation
supplementary/     supplementary tables and notes (to be added)
```

## Data

The genome assemblies and annotations come from the JGI MycoCosm *1000 Fungal
Genomes* initiative and are **not** redistributed here. The exact set used is
listed in [`docs/genome_list.tsv`](docs/genome_list.tsv) — 735 portal IDs with
their phylum, subphylum, class, order and family. One assembly is dropped by
QC, leaving the 734 genomes reported in the paper.

Download the assemblies and GFF3 annotations from MycoCosm by portal ID, then
run the scripts in [`data_prep/`](data_prep/).

## Known limitations

- **Scripts contain absolute paths from the authors' machine.** Paths rooted at
  the original working directory appear in some files, mostly pointing at
  dataset and checkpoint locations. They have deliberately not been rewritten,
  so that the code matches what actually produced the reported numbers. You
  will need to adjust them for your own layout. Grep for `/home/` to find them.
- **Derived datasets are not yet released.** A deposit is planned. For now,
  rebuild them from the JGI source with `data_prep/`.

## License

[MIT](LICENSE).

## Citation

Citation details will be added once the paper is published.
