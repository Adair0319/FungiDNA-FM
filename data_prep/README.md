# Pretraining data preparation

This directory contains the scripts that turn a MycoCosm download into the
pretraining corpus used by FungiDNA-FM. It covers **pretraining only**.

## What is here

```
genomes/
  copy_assembly.sh    collect the genome assemblies from a MycoCosm download tree
  copy_gff3.sh        collect the GFF3 annotations from the same tree
pretrain/
  prepare_pretrain_data.py   filter contigs, collect long contigs, train the
                             Unigram tokenizer, index the corpus
```

### 1. Collect the assemblies

MycoCosm delivers one directory per genome. The copy scripts take the download
root and a destination directory (both explicit — nothing defaults to an
author-specific path):

```bash
bash data_prep/genomes/copy_assembly.sh /path/to/mycocosm_download assembly_genome
bash data_prep/genomes/copy_gff3.sh      /path/to/mycocosm_download gff3
```

`copy_assembly.sh` keeps the first `*.fasta.gz` per genome directory;
`copy_gff3.sh` selects the annotation file from `Filtered_Models___best__/`.
Both write a `copy_log.txt` recording what was copied or skipped. They depend on
the MycoCosm directory layout, not on any local convention.

### 2. Build the corpus

```bash
python data_prep/pretrain/prepare_pretrain_data.py \
    --raw_dir        /path/to/raw_genomes \
    --taxonomy_xlsx  /path/to/fungi_classification.xlsx \
    --output_dir     data/processed \
    --phase1_min_len 5000 \
    --phase2_min_len 50000 \
    --max_n_frac     0.10
```

This applies the quality control described in Supplementary Note S1 (contigs of
at least 5 kb, at most 10 % ambiguous bases), keeps long contigs for Phase 2,
trains the 4,096-token SentencePiece Unigram tokenizer on a 5 % sample of the
filtered sequence lines, and writes the corpus index consumed by
`fungidna/data/pretrain_dataset.py`.

The trained tokenizer is distributed with the model weights
(`tokenizer/bpe_fungi.model`, see `docs/weights_manifest.md`), so training a new
one is only needed for a new corpus.

## What is not here

- **Downstream dataset construction.** The builders for the five evaluated
  datasets (taxonomy windows, CDS/intergenic, short-CDS, splice sites, BGC) are
  not part of this repository. `Supplementary Note S3` describes the procedures;
  the code is not released.
- **Genome data.** Assemblies and annotations come from JGI MycoCosm and are not
  redistributed. `docs/genome_list.tsv` lists the 735 portal IDs.
- **Annotation post-processing.** The scripts that reshaped or filtered the
  raw MycoCosm annotation directories were one-off operational tools and are
  not included.
