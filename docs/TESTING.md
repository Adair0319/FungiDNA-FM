# Testing

## Running the suite

```bash
python -m unittest discover -s tests            # everything
python -m unittest tests.test_representation -v # one module
```

The model tests need **CUDA** (the backbone uses `flash-attn` and `mamba-ssm`
kernels; there is no CPU path) and the **real checkpoints**. Point the tests at
them, or leave them unset to use the authors' local layout:

```bash
export FUNGIDNA_TEST_TOKENIZER=/path/to/bpe_fungi.model
export FUNGIDNA_TEST_UNIGRAM_BACKBONE=/path/to/unigram/backbone_final.pt
export FUNGIDNA_TEST_SNT_BACKBONE=/path/to/snt/backbone_final.pt
export FUNGIDNA_TEST_WEIGHTS=/path/to/checkpoint/root   # for the task tests
```

Tests **skip** (they never fall back to random weights) when CUDA or a
checkpoint is missing. `tests/test_downstream_tasks.py` writes a temporary
weights config with a `unigram:` and `snt:` section, so it exercises the
variant-scoped loader end to end.

## What is covered

| Module | Covers |
|---|---|
| `test_core.py` | tokenizer id mapping, model config, task-name normalisation, checkpoint shape checking, label maps |
| `test_representation.py` | tokenizer contract, per-variant backbone resolution, real-checkpoint loading, single/batch shapes, padding invariance, no-CLS pooling, absence of a training-module dependency |
| `test_downstream_tasks.py` | all five tasks × both variants × (taxonomy) all five ranks, against the real checkpoints |
| `test_weight_loader.py` | registry → env → YAML precedence, malformed config handling |
| `test_sequence.py`, `test_validation.py`, `test_api.py`, `test_report.py`, `test_dispatcher.py` | I/O, validation gate, report format, dispatch |

Current status: **55 tests, all passing.**

## Head and pooling per variant (what the handlers reproduce)

| Task | Unigram | SNT |
|---|---|---|
| `cds`, `short_cds`, `splice_site` | `ClassificationHead(hidden_dim=256)` over `hidden[:, 0]` | `MeanPoolHead(hidden_dim=128)` over `hidden[:, 1:].mean` |
| `bgc_boundary` | `MeanPoolHead(hidden_dim=128)` over `hidden[:, 1:].mean` | same |
| `taxonomy` | Linear+BatchNorm MLP over `mean(all positions)` | LayerNorm+Linear MLP over `hidden[:, 1:].mean` |
| `embedding` (Representation) | masked mean over DNA tokens, **no CLS** | same, single-nucleotide tokens |

The `[CLS]` conventions belong to the task heads; the Representation API never
adds a CLS token.

## Verification status of the dependency set

`requirements.txt` / `environment.yml` pin the combination below. What was
actually checked, and where:

| Item | Status |
|---|---|
| `torch 2.5.1` + `triton 3.1.0` + CUDA 12.1 | verified in a **fresh** conda env |
| `mamba-ssm 2.3.1` (source build) | verified: imports, and `Mamba3` is absent so the `triton.set_allocator` failure cannot occur |
| `causal-conv1d 1.6.2.post1` (source build) | verified: Mamba2 block forward runs |
| `transformers 4.47.0` | verified as required — `mamba-ssm` 2.3.x imports `GenerateDecoderOnlyOutput`, removed in 5.x |
| `flash-attn 2.8.3.post1` | **not verified in a fresh env this round** — the source build was interrupted. It is known to work in the authors' existing environment (`flash-attn 2.8.3.post1`, same torch/triton). |
| End-to-end model tests in a fresh env | **not completed**, because they need `flash-attn` |

All 55 tests do pass in the authors' existing environment
(`torch 2.5.1`, `triton 3.1.0`, `mamba-ssm 2.3.2.post1` with a local
`__init__.py` patch, `flash-attn 2.8.3.post1`, `transformers 4.47.0`).
That environment is **not** reproducible from `requirements.txt` precisely
because of the `mamba-ssm` patch; the pins above are the reproducible
alternative that avoids patching any third-party file.

## Known limitations

- **CUDA only.** No CPU fallback.
- **Long sequences** are limited by GPU memory: the backbone is ~113 M
  parameters, but attention and the Mamba scan state scale with length;
  10 kb windows are the intended use.
- **Weights are not in this repository.** See `docs/weights_manifest.md` for
  the published paths, SHA256 and publication status.
- **Single-fold checkpoints.** The released task models are one fold each
  (fold 4 for `cds`/`short_cds`/`splice_site`/`bgc_boundary`, fold 1 for
  `taxonomy`); the paper's five-fold mean ± SD is a different aggregate.
- **BGC windows in the demo** are shorter than the 10 kb training window; the
  model still runs, but predictions are only meaningful for 10 kb input.
