# Model weights manifest

This file lists the exact checkpoint files to upload to Hugging Face for
inference with the `fungidna.invoke` handlers. The handlers load these via the
`WeightLoader` (env vars or `~/.config/fungidna/weights.yaml`), and each load is
validated with a matched-key count so a layout mismatch fails loudly.

All source paths are relative to the original working tree
`fungi_dna_model/`. Sizes are as measured on disk.

## Recommended Hugging Face layout

```
<user>/FungiDNA-FM
├── backbone_final.safetensors   (or backbone_final.pt)   # pretrained backbone
├── tokenizer/
│   ├── bpe_fungi.model
│   └── bpe_fungi.vocab
├── coding_potential/
│   └── best_model.pt            # ours_finetune fold_0
├── splice_site/
│   └── best_model.pt            # fullft_lr2e-5 fold_0
├── taxonomy/
│   ├── phylum/mlp_best.pt
│   ├── subphylum/mlp_best.pt
│   ├── class/mlp_best.pt
│   ├── order/mlp_best.pt
│   └── family/mlp_best.pt
└── bgc_boundary/
    └── best_model.pt            # ours_ft
```

## Files to upload

| Purpose | Source file | Size | HF destination |
|---|---|---|---|
| Pretrained backbone (shared) | `checkpoints/phase2_joint/backbone_final.pt` | 217 MB | `backbone_final.pt` |
| SentencePiece tokenizer model | `data/processed/bpe_fungi.model` | 294 KB | `tokenizer/bpe_fungi.model` |
| SentencePiece vocab | `data/processed/bpe_fungi.vocab` | 63 KB | `tokenizer/bpe_fungi.vocab` |
| Coding potential (ours, fine-tuned) | `checkpoints/cds_intergenic/ours_finetune/fold_0/best_model.pt` | 227 MB | `coding_potential/best_model.pt` |
| Splice site (ours, full fine-tune LR 2e-5) | `checkpoints/splice_v2/fullft_lr2e-5/fold_0/best_model.pt` | 434 MB | `splice_site/best_model.pt` |
| Taxonomy head — phylum | `checkpoints/phase2_joint_eval_final/phylum/fold1/mlp_best.pt` | 0.8 MB | `taxonomy/phylum/mlp_best.pt` |
| Taxonomy head — subphylum | `checkpoints/phase2_joint_eval_final/subphylum/fold1/mlp_best.pt` | ~0.9 MB | `taxonomy/subphylum/mlp_best.pt` |
| Taxonomy head — class | `checkpoints/phase2_joint_eval_final/class/fold1/mlp_best.pt` | ~1 MB | `taxonomy/class/mlp_best.pt` |
| Taxonomy head — order | `checkpoints/phase2_joint_eval_final/order/fold1/mlp_best.pt` | ~2 MB | `taxonomy/order/mlp_best.pt` |
| Taxonomy head — family | `checkpoints/phase2_joint_eval_final/family/fold1/mlp_best.pt` | ~3 MB | `taxonomy/family/mlp_best.pt` |
| BGC boundary (ours, fine-tuned) | `checkpoints/bgc_boundary/ours_ft/best_model.pt` | 227 MB | `bgc_boundary/best_model.pt` |

## Notes

- **Fold selection.** Only one fold per task is needed for inference. For the
  five-fold-CV checkpoints we pick `fold_0` (or, where a `cv_summary.json`
  records `best_fold`, that fold). Intermediate fold weights are not required.
- **Taxonomy head only.** The taxonomy MLP head (`mlp_best.pt`) is a bare
  `net.*` state dict and does NOT include the backbone — the backbone is the
  shared `backbone_final.pt` resolved separately. The per-rank class counts are:
  phylum 6, subphylum 11, class 26, order 98, family 281.
- **Do not upload** the 23 intermediate `phase2_joint/backbone-*.pt` snapshots
  (217 MB each) or the exploratory model variants under `splice_v2/`
  (dnabert2, ntv2, transformer, onehot, randfull, LR sweeps) and `bgc_boundary/`
  (gena, cnn, kmer, mamba2, mean).
- **dtype** — backbone / coding / bgc are bfloat16; splice is float32. The
  loaders cast the model to the checkpoint dtype automatically.
- **safetensors** — converting `backbone_final.pt` to safetensors is recommended
  for Hugging Face (smaller, faster, safer). The `load_checkpoint` helper reads
  `.pt` via `torch.load(weights_only=True)`; safetensors loading is not wired in
  yet, so keep a `.pt` alongside if you convert.

## Configuring the handler to find these

`~/.config/fungidna/weights.yaml`:

```yaml
backbone: /path/to/backbone_final.pt
tokenizer: /path/to/bpe_fungi.model
coding_potential: /path/to/coding_potential/best_model.pt
splice_site: /path/to/splice_site/best_model.pt
taxonomy: /path/to/taxonomy/phylum/mlp_best.pt
bgc_boundary: /path/to/bgc_boundary/best_model.pt
```

Or set `FUNGIDNA_BACKBONE_WEIGHTS`, `FUNGIDNA_TOKENIZER_WEIGHTS`,
`FUNGIDNA_CODING_POTENTIAL_WEIGHTS`, etc. (see `fungidna/invoke/handlers/base.py`).
