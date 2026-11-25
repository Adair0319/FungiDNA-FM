# Weights manifest

Checkpoints used by the released models. The path column gives the layout inside
the distributed archive; nothing here is stored in this Git repository.

> **Distribution.** The weights are distributed through Google Drive:
> **download link: To be added.** No file below is currently available from a
> public URL. Names, sizes and SHA256 are recorded here so a download can be
> verified once the archive is published.

## Backbones

| Variant | Path | Bytes | SHA256 | Checkpoint |
|---|---|---|---|---|
| unigram | `unigram/backbone.pt` | 226,891,524 | `e20d664c1ab2b0ee73d26f552bb6ca44206060a3fa9b60e756d2b06aabb3c2be` | final Phase 2 checkpoint, step 16,000 (early stop) |
| snt | `snt/backbone.pt` | 226,898,729 | `c35b337bf5e742530cf19b633bd047010c1d8af924bf9c41f612b37cc663ec69` | final Phase 2 checkpoint, step 33,000 (early stop) |

## Taxonomy (frozen backbone + per-rank MLP probe, fold 1)

| Variant | Rank | Path | SHA256 |
|---|---|---|---|
| unigram | phylum | `unigram/tasks/taxonomy/phylum/mlp_best.pt` | `98a16d9ee5bff3d5de2d1b951f5ccef468888fbf64664f7b55c0ae640e36fdaf` |
| unigram | subphylum | `unigram/tasks/taxonomy/subphylum/mlp_best.pt` | `207817e2131bba680ea9b6e0858c7fb8b5d75421d84404fa417472079ee3958a` |
| unigram | class | `unigram/tasks/taxonomy/class/mlp_best.pt` | `5c2b062cfa71e5b271b674fcd4818d78c64d28c0bd5668a68d685e530ee384f9` |
| unigram | order | `unigram/tasks/taxonomy/order/mlp_best.pt` | `db007b8ac813b5b1cdf9da2c1a321272b7c814a7c5a1399d8413b60d6670ed8d` |
| unigram | family | `unigram/tasks/taxonomy/family/mlp_best.pt` | `4f1d023ad628e06c2ad4f98f13319b66ff76be55029bddabe5b983070710cd90` |
| snt | phylum | `snt/tasks/taxonomy/phylum/mlp_best.pt` | `ca7bdbb15e2816b1c1133f0f3e410a0fad3e9dee5f8385fd3baad7e55991cb08` |
| snt | subphylum | `snt/tasks/taxonomy/subphylum/mlp_best.pt` | `a172ee719405ea92cdaa116ac0b0ea72148a21107f975381071b158bd46fb2b1` |
| snt | class | `snt/tasks/taxonomy/class/mlp_best.pt` | `b5ffc08a81b64d172dd24f21d5904844f1c4f881701181093d9f5aa790ee323f` |
| snt | order | `snt/tasks/taxonomy/order/mlp_best.pt` | `f58b08231a9ac99c404d09fecaca8381e3e7a2564ea90a8d343b383fef71e8a1` |
| snt | family | `snt/tasks/taxonomy/family/mlp_best.pt` | `22ab24e3eaec8374efa007a9bc59d45eb6d1a55a27b5e264bf1c24ce3fee122c` |

Fold 1 is one of the five cross-validation folds, not a fold chosen for being the
best performer; the paper's five-fold mean ± SD is a separate aggregate.

## Downstream fine-tuned task models

| Variant | Task | Path | Bytes | SHA256 | Fold | Training data |
|---|---|---|---|---|---|---|
| unigram | cds | `unigram/tasks/cds/best_model.pt` | 227,287,523 | `ecf4027ef5aaecb471b43efd87fc35be30421abff920ea1185546e17d7fc2de3` | fold 0 | `task_cds_intergenic_matched_v2_50k` |
| snt | cds | `snt/tasks/cds/best_model.pt` | 227,090,147 | `8b42ea311613c37e161a63c028d13fc9508dd8c54d037a793125576dbede9e9b` | fold 0 | `task_cds_intergenic_matched_v2_50k` |
| unigram | short_cds | `unigram/tasks/short_cds/best_model.pt` | 227,301,667 | `5e7a14dc007a8ac725ca8f044b3b3f7912a8bcae97d5137411b6c7b08ee3bd63` | fold 4 | `task_cds_intergenic_hard_v2_noeasy` |
| snt | short_cds | `snt/tasks/short_cds/best_model.pt` | 227,104,291 | `5095addf3a9ea30209759343486b266ced5a2e8cf6663d0043d4bc31af753310` | fold 4 | `task_cds_intergenic_hard_v2_noeasy` |
| unigram | splice_site | `unigram/tasks/splice_site/best_model.pt` | 227,302,179 | `add4ed7f0c6ff0e1c126c8efd8c743cada7d2658845cdf0cc13694c858fb5dd1` | fold 4 | `task2_splice_site_v3_canonical_20k` |
| snt | splice_site | `snt/tasks/splice_site/best_model.pt` | 227,104,547 | `473b13f7d4fa00af58ede961ca0994eb23e8e4053ceeb269e23268eda93df2b6` | fold 4 | `task2_splice_site_v3_canonical_20k` |
| unigram | bgc_boundary | `unigram/tasks/bgc_boundary/best_model.pt` | 227,104,355 | `de870ff07f8942782dc60ab23f485f23249620201f3e10608db1800b4c626aef` | fold 4 | FunBGC 10 kb (`task_bgc_v2`, 702 windows / 472 species) |
| snt | bgc_boundary | `snt/tasks/bgc_boundary/best_model.pt` | 227,104,355 | `f0b187c94ccbd857fe142d8f83fb96062d787554dad1ded9b96ca0dec10fde0d` | fold 4 | FunBGC 10 kb (`task_bgc_v2`, 702 windows / 472 species) |

Notes:

- `short_cds`, `splice_site` and `bgc_boundary` are the best-validation
  checkpoints of the last fold (fold 4); the earlier folds' files were
  overwritten during training. `cds` ships its fold-0 checkpoint (verified by
  SHA256 against the training output, not assumed from the file name).
- BGC models use species-grouped five-fold splits (seed 42); folds share one
  `fold_ids.json`, so no species appears in more than one fold.
- Each released file is a single-fold model. The paper's five-fold mean ± SD is
  an aggregate over five such models and is not the performance of one file.
- Checkpoints from earlier, differently-named task releases are not part of this
  release.

## Tokenizer

| Path | Bytes | SHA256 |
|---|---|---|
| `tokenizer/bpe_fungi.model` (SentencePiece Unigram, vocab 4096) | 300,706 | `4af11308498dc8f4c8f80dc5fee3231eb3e8be8a4a998e58a4fd003a0522d9b7` |
| `tokenizer/bpe_fungi.vocab` | 64,104 | `4eff286e38440d13a9d78c6ea190e877cdd1bf5ca17a642e1e574ded42af045d` |

The file name `bpe_fungi.model` is historical: the tokenizer is a SentencePiece
**Unigram** model. The Single-Nucleotide variant needs no tokenizer file.

## Provenance and training protocol

Each released file was matched to its training output by SHA256 (the training
tree is not part of this repository). The protocol column records what the file
was actually trained with, which is not the same for every task.

| File | Source run directory | Protocol |
|---|---|---|
| `unigram/backbone.pt`, `snt/backbone.pt` | `checkpoints/phase2_joint/backbone_final.pt`, `checkpoints/phase2_joint_single_nt/backbone_final.pt` | Phase 2 of Note S2 (MLM + SupCon) |
| `*/tasks/cds/best_model.pt` | `checkpoints/extended_finetune_cds_intergenic_v2_50k/{fungidna_bpe,fungidna_snt}/full/fold_0/` | AdamW 5 × 10⁻⁵, ReduceLROnPlateau, early stopping on validation macro-F1 (patience 10, ≤ 40 epochs) — the Note S5 protocol |
| `*/tasks/short_cds/best_model.pt` | `checkpoints/sorf_v3/sorf_{bpe,snt}_finetune/` | **5 epochs, 10 % warm-up followed by cosine decay; no plateau, no F1 early stopping.** Differs from Note S5. Five-fold mean macro-F1 0.705 (Unigram) and 0.703 (SNT) |
| `*/tasks/splice_site/best_model.pt` | `checkpoints/splice_v3/{bpe_ft,snt_ft}/` | Same 5-epoch warm-up + cosine schedule. Five-fold mean macro-F1 0.687 (Unigram) and 0.587 (SNT), versus the Note S5 protocol's 0.858 and 0.775 reported in Table S2 |
| `*/tasks/bgc_boundary/best_model.pt` | `checkpoints/bgc_boundary_v2/funbgc10_{bpe,snt}_ft/` | Note S5 BGC protocol (10 % warm-up + cosine, early stopping on validation loss, ≤ 30 epochs) |
| `*/tasks/taxonomy/{rank}/mlp_best.pt` | Unigram: `checkpoints/phase2_joint_eval_final/{rank}/fold1/`; SNT: `checkpoints/taxonomy_single_nt_retrain/{rank}/fold1/` | Unigram: BatchNorm probe, focal loss, AdamW 1 × 10⁻⁴. SNT: LayerNorm probe, cross-entropy, AdamW 1 × 10⁻³ |
