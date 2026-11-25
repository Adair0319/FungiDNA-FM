# Testing

## Running the suite

Run everything **from the repository root**:

```bash
cd FungiDNA-FM
python -m unittest discover -s tests            # everything
python -m unittest tests.test_representation -v # one module
```

### Why the working directory matters

This repository has **no packaging metadata** (`pyproject.toml`, `setup.py` or
`setup.cfg`), so `pip install -e .` is **not** supported. The `fungidna` package
is importable only when the repository root is on `sys.path` — which is the case
when you run Python from that directory:

```bash
cd FungiDNA-FM
python -c "from fungidna.representation import RepresentationModel; print('ok')"
```

From any other directory, either set `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/FungiDNA-FM:$PYTHONPATH
```

or reuse the bootstrap already in the examples
(`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`).

### Real checkpoints

The model tests need **CUDA** (the backbone uses `flash-attn` and `mamba-ssm`
kernels; there is no CPU path) and the **real checkpoints**:

```bash
export FUNGIDNA_TEST_TOKENIZER=/path/to/bpe_fungi.model
export FUNGIDNA_TEST_UNIGRAM_BACKBONE=/path/to/unigram/backbone.pt
export FUNGIDNA_TEST_SNT_BACKBONE=/path/to/snt/backbone.pt
export FUNGIDNA_TEST_WEIGHTS=/path/to/checkpoint/root   # for the task tests
```

`FUNGIDNA_TEST_WEIGHTS` is the root of the authors' checkpoint tree; the module
resolves paths such as `checkpoints/phase2_joint/backbone_final.pt` beneath it.
If your download uses the archive layout (`unigram/backbone.pt`, ...), edit the
paths at the top of `tests/test_downstream_tasks.py`.

Tests **skip** — they never fall back to random weights — when CUDA or a
checkpoint is missing. `tests/test_downstream_tasks.py` writes a temporary
weights config with a `unigram:` and `snt:` section, so it exercises the
variant-scoped loader end to end.

## What is covered

The suite has 16 modules:

| Module | Covers |
|---|---|
| `test_core.py` | tokenizer id mapping, model config, task-name normalisation, checkpoint shape checking, label maps |
| `test_representation.py` | tokenizer contract, per-variant backbone resolution, real-checkpoint loading, single/batch shapes, padding invariance, layer-8 no-CLS pooling, values finite, absence of a training-module dependency |
| `test_downstream_tasks.py` | all five tasks × both variants × (taxonomy) all five ranks, against the real checkpoints |
| `test_natural_language.py` | six-task prompt parsing, CLI flags, one real end-to-end CLI run |
| `test_parser_variant_rank.py` | variant/rank parsing, ambiguity and unsupported-rank errors, path forms, backward compatibility, CLI precedence |
| `test_tokenizer_helpers.py` | `token_type_for` / `encode_window` / `mlm_vocab_size`, single source of truth for the variant map, integer-API regression |
| `test_snt_pretraining.py` | MLM head tying per variant, dataset encoding and masking vocabulary, effective batch per Phase 2 config |
| `test_phase1_early_stop.py` | evaluation-node rules, two-process Gloo stop broadcast, non-distributed passthrough |
| `test_documented_deviations.py` | the two documented model-level deviations from the paper: per-head (non-positional) rotary embedding, LayerNorm after the input embedding |
| `test_bgc_model.py` | BGC boundary model construction |
| `test_weight_loader.py` | registry → env → YAML precedence, malformed config handling |
| `test_sequence.py`, `test_validation.py`, `test_api.py`, `test_report.py`, `test_dispatcher.py` | I/O, validation gate, report format, dispatch |

The test suite currently contains **136 unit tests**, all passing in the
authors' environment (see the verification table below).

### What the task tests assert, and what they do not

- They assert that each handler runs on a single sequence and returns the
  documented output type and value range.
- `load_checkpoint` raises on any unexpected shape mismatch (the Unigram
  `nt_embed` mismatch is the single explicit exception), so a passing run means
  every checkpoint tensor matched the model's shapes.
- They do **not** assert `missing == 0` / `unexpected == 0` per task; those are
  printed by `load_checkpoint` but not asserted. A separate static key/shape
  comparison of all 18 task-head combinations reported 0 missing, 0 unexpected
  and 0 shape mismatches.
- `test_downstream_tasks.py` does **not** assert that the BGC regression outputs
  are finite (`isinstance(float)` accepts NaN/inf). Finiteness is asserted only
  for the Representation embeddings.

## Head and pooling per variant (what the handlers reproduce)

| Task | Unigram | SNT |
|---|---|---|
| `cds`, `short_cds`, `splice_site` | `ClassificationHead(hidden_dim=256)` over `hidden[:, 0]` | `MeanPoolHead(hidden_dim=128)` over `hidden[:, 1:].mean` |
| `bgc_boundary` | `MeanPoolHead(hidden_dim=128)` over `hidden[:, 1:].mean` | same |
| `taxonomy` | Linear+BatchNorm MLP over `mean(all positions)` | LayerNorm+Linear MLP over `hidden[:, 1:].mean` |
| `embedding` (Representation) | masked mean over DNA tokens at the output of **layer 8 of 24** (the attention block: attention sub-layer + SwiGLU), before `final_norm`, **no CLS** | same layer and pooling, single-nucleotide tokens |

The `[CLS]` conventions belong to the task heads; the Representation API never
adds a CLS token.

## Command line interface

`python -m fungidna "<request>"` recognises all six window-level tasks:
**embedding, taxonomy, CDS/intergenic, short-CDS, splice-site and BGC boundary**.
The same text may also name the variant and, for taxonomy, the rank.

- Variant: `unigram`, `snt` or `single-nucleotide`; defaults to `unigram`.
- Taxonomy ranks: `phylum`, `subphylum`, `class`, `order`, `family`; defaults to
  `phylum`. Levels the project does not model (`genus`, `species`, `kingdom`, …)
  are rejected with an explicit error.
- Naming two different variants, or two different ranks, in one request is
  rejected as ambiguous.
- Non-taxonomy tasks carry no rank; an explicit `--rank` on such a request
  prints a note that the flag is ignored.
- `--variant` / `--rank` flags take precedence over the request text.
- Paths containing spaces must be quoted; relative paths are accepted and
  normalised by the validation layer.

```bash
python -m fungidna "embedding of x.fasta using SNT"
python -m fungidna "classify taxonomy at family of genome.fasta using SNT"
python -m fungidna "predict splice sites in data/sample.fasta" --variant unigram
```

The same tasks are available from Python with structured arguments:

```python
from fungidna.invoke import run_task
run_task(file_path="seq.fasta", task="short_cds", variant="snt")
run_task(file_path="genome.fasta", task="taxonomy", variant="unigram", rank="family")
```

## Verification status of the dependency set

`requirements.txt` / `environment.yml` pin the combination below. What was
actually checked, and where:

| Item | Status |
|---|---|
| `torch 2.5.1` + `triton 3.1.0` + CUDA 12.1 | verified in a **fresh** conda env |
| `mamba-ssm 2.3.1` (source build) | verified: imports, and `Mamba3` is absent so the `triton.set_allocator` failure cannot occur |
| `causal-conv1d 1.6.2.post1` (source build) | verified: Mamba2 block forward runs |
| `transformers 4.47.0` | verified as required — `mamba-ssm` 2.3.x imports `GenerateDecoderOnlyOutput`, removed in 5.x |
| `flash-attn 2.8.3.post1` | **not verified in a fresh env** — the source build was interrupted. It is known to work in the authors' existing environment. |
| End-to-end model tests in a fresh env | **not completed**, because they need `flash-attn` |
| `environment.yml` as a whole | **not verified**: the file has never been consumed by a single `conda env create`. The `pip: -r requirements.txt` entry is resolved **relative to the directory you run `conda env create` from**, so it must be executed at the repository root. |

All 136 tests pass in the authors' existing environment
(`torch 2.5.1`, `triton 3.1.0`, `mamba-ssm 2.3.2.post1` with a local
`__init__.py` patch, `flash-attn 2.8.3.post1`, `transformers 4.47.0`).
That environment is **not** reproducible from `requirements.txt` precisely
because of the `mamba-ssm` patch; the pins above are the reproducible
alternative that avoids patching any third-party file, but the `flash-attn`
part of it is still unverified.

## Phase 1 checkpoint semantics

- `checkpoint-<step>.pt` — the state at that training step (optimizer,
  scheduler and step included); written every `save_every` steps.
- `final_model.pt` — the state at the moment training ended (max steps or early
  stop).
- There is **no separate best-model checkpoint**: `final_model.pt` is not
  guaranteed to be the lowest-validation-loss checkpoint, and the early-stop
  metric is the running training loss (no separate validation forward).
- Early stopping is synchronised across DDP ranks: rank 0 decides, the decision
  is broadcast at every evaluation node, and all ranks leave the loop at the
  same step. A non-positive `early_stop_patience` disables early stopping.

## Known limitations

- **CUDA only.** No CPU fallback.
- **Long sequences** are limited by GPU memory: the backbone is ~113 M
  parameters, but attention and the Mamba scan state scale with length;
  10 kb windows are the intended use.
- **Weights are not in this repository.** They are distributed through Google
  Drive; the download link is `To be added`. See `docs/weights_manifest.md` for
  names, sizes and SHA256.
- **Single-fold checkpoints.** The released task models are one fold each
  (fold 4 for `cds`/`short_cds`/`splice_site`/`bgc_boundary`, fold 1 for
  `taxonomy`); the paper's five-fold mean ± SD is a different aggregate.
- **BGC windows in the demo** are shorter than the 10 kb training window; the
  model still runs, but predictions are only meaningful for 10 kb input.
- **BGC outputs are not checked for NaN/inf** by the tests.
