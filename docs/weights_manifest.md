# Weights manifest

> **Status: candidate weights.** These are the checkpoint files currently under
> consideration for public release. Unless a row is explicitly marked
> "confirmed", the training-step / fold / dataset attribution has **not** been
> formally approved for the paper and must be treated as a candidate, not a
> final release weight. Weights are hosted on Hugging Face, not in this
> repository.

## Backbones

| variant | file | bytes | SHA256 | step | strategy | notes |
|---|---|---|---|---|---|---|
| unigram | `phase2_joint/backbone_final.pt` | 226,891,524 | `e20d664c1ab2b0ee73d26f552bb6ca44206060a3fa9b60e756d2b06aabb3c2be` | **unconfirmed** (do not cite 16,000) | Phase 2 MLM+SupCon | final step not yet determined |
| snt | `phase2_joint_single_nt/backbone_final.pt` | 226,898,729 | `c35b337bf5e742530cf19b633bd047010c1d8af924bf9c41f612b37cc663ec69` | 33,000 | Phase 2 MLM+SupCon | early stop at 33,000 |

## Taxonomy (frozen backbone + per-rank MLP probe, fold 1)

| variant | rank | file | SHA256 |
|---|---|---|---|
| unigram | phylum | `phase2_joint_eval_final/phylum/fold1/mlp_best.pt` | `98a16d9ee5bff3d5de2d1b951f5ccef468888fbf64664f7b55c0ae640e36fdaf` |
| unigram | subphylum | `phase2_joint_eval_final/subphylum/fold1/mlp_best.pt` | `207817e2131bba680ea9b6e0858c7fb8b5d75421d84404fa417472079ee3958a` |
| unigram | class | `phase2_joint_eval_final/class/fold1/mlp_best.pt` | `5c2b062cfa71e5b271b674fcd4818d78c64d28c0bd5668a68d685e530ee384f9` |
| unigram | order | `phase2_joint_eval_final/order/fold1/mlp_best.pt` | `db007b8ac813b5b1cdf9da2c1a321272b7c814a7c5a1399d8413b60d6670ed8d` |
| unigram | family | `phase2_joint_eval_final/family/fold1/mlp_best.pt` | `4f1d023ad628e06c2ad4f98f13319b66ff76be55029bddabe5b983070710cd90` |
| snt | phylum | `taxonomy_single_nt_retrain/phylum/fold1/mlp_best.pt` | `ca7bdbb15e2816b1c1133f0f3e410a0fad3e9dee5f8385fd3baad7e55991cb08` |
| snt | subphylum | `taxonomy_single_nt_retrain/subphylum/fold1/mlp_best.pt` | `a172ee719405ea92cdaa116ac0b0ea72148a21107f975381071b158bd46fb2b1` |
| snt | class | `taxonomy_single_nt_retrain/class/fold1/mlp_best.pt` | `b5ffc08a81b64d172dd24f21d5904844f1c4f881701181093d9f5aa790ee323f` |
| snt | order | `taxonomy_single_nt_retrain/order/fold1/mlp_best.pt` | `f58b08231a9ac99c404d09fecaca8381e3e7a2564ea90a8d343b383fef71e8a1` |
| snt | family | `taxonomy_single_nt_retrain/family/fold1/mlp_best.pt` | `22ab24e3eaec8374efa007a9bc59d45eb6d1a55a27b5e264bf1c24ce3fee122c` |

> fold 1 is **not** the best-performing fold; it is one of the five folds. The
> five-fold mean ± std reported in the paper is a separate aggregate.

## Downstream fine-tuned task models

| variant | task | file | bytes | SHA256 | fold | dataset (confirmed) |
|---|---|---|---|---|---|---|
| unigram | cds | `extended_finetune_cds_intergenic_v2_50k/fungidna_bpe/full/fold_0/best_model.pt` | 227,287,523 | `ecf4027ef5aaecb471b43efd87fc35be30421abff920ea1185546e17d7fc2de3` | fold 0 | `task_cds_intergenic_matched_v2_50k` |
| snt | cds | `extended_finetune_cds_intergenic_v2_50k/fungidna_snt/full/fold_0/best_model.pt` | 227,090,147 | `8b42ea311613c37e161a63c028d13fc9508dd8c54d037a793125576dbede9e9b` | fold 0 | `task_cds_intergenic_matched_v2_50k` |
| unigram | short_cds | `sorf_v3/sorf_bpe_finetune/best_model.pt` | 227,301,667 | `5e7a14dc007a8ac725ca8f044b3b3f7912a8bcae97d5137411b6c7b08ee3bd63` | fold 4 | `task_cds_intergenic_hard_v2_noeasy` |
| snt | short_cds | `sorf_v3/sorf_snt_finetune/best_model.pt` | 227,104,291 | `5095addf3a9ea30209759343486b266ced5a2e8cf6663d0043d4bc31af753310` | fold 4 | `task_cds_intergenic_hard_v2_noeasy` |
| unigram | splice_site | `splice_v3/bpe_ft/best_model.pt` | 227,302,179 | `add4ed7f0c6ff0e1c126c8efd8c743cada7d2658845cdf0cc13694c858fb5dd1` | fold 4 | `task2_splice_site_v3_canonical_20k` |
| snt | splice_site | `splice_v3/snt_ft/best_model.pt` | 227,104,547 | `473b13f7d4fa00af58ede961ca0994eb23e8e4053ceeb269e23268eda93df2b6` | fold 4 | `task2_splice_site_v3_canonical_20k` |
| unigram | bgc_boundary | `bgc_boundary_v2/funbgc10_bpe_ft/best_model.pt` | 227,104,355 | `de870ff07f8942782dc60ab23f485f23249620201f3e10608db1800b4c626aef` | single | **unconfirmed** |
| snt | bgc_boundary | `bgc_boundary_v2/funbgc10_snt_ft/best_model.pt` | 227,104,355 | `f0b187c94ccbd857fe142d8f83fb96062d787554dad1ded9b96ca0dec10fde0d` | single | **unconfirmed** |

> - `short_cds` and `splice_site` use the **fold 4** best-validation checkpoint
>   (the last fold, not a best-of-five selection). Folds 0–3 were overwritten.
> - `bgc_boundary` data source (FunBGC 10 kb vs MIBiG vs 20 kb window) is
>   **unconfirmed**; the checkpoint is saved as `{"model", "epoch", "val_loss"}`.
> - None of these single-fold checkpoints represent the five-fold mean ± std.

## Tokenizer

| file | bytes | SHA256 |
|---|---|---|
| `data/processed/bpe_fungi.model` (SentencePiece Unigram, vocab 4096) | 300,706 | `4af11308498dc8f4c8f80dc5fee3231eb3e8be8a4a998e58a4fd003a0522d9b7` |
| `data/processed/bpe_fungi.vocab` | 64,104 | `4eff286e38440d13a9d78c6ea190e877cdd1bf5ca17a642e1e574ded42af045d` |
