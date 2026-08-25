# Experiment 1 — Analysis of pretrained representations

**Produces:** Figure 2, Supplementary Table S1 (Methods 2.6)

Unsupervised probe of what the frozen backbone encodes. Mean-pooled 768-dim
representations → PCA (50 dims) → UMAP → K-means, scored against known labels
with ARI, NMI, homogeneity, completeness and V-measure. A 4-mer frequency
representation is the baseline.

## Scripts

| Script | Produces |
|---|---|
| `umap_clustering_phylum_depth.py` | Fig 2A–D — phylum clustering (88,172 sequences, six phyla), swept across the six hybrid blocks |
| `umap_clustering_elements.py` | Fig 2E–G — genomic-element clustering (100,000 sequences: CDS, intergenic, 5′ UTR, 3′ UTR, intron) |
| `checkpoint_ablation.py` | Fig 2D, 2H — Phase-1 vs Phase-2 checkpoint comparison |
| `phase1_ablation.py`, `fix_elements_ablation.py` | supporting ablations |
| `run_elements_umap.sh` | launcher for the element analysis |

## Reported values

Phylum, Block 1: ARI 0.5497, NMI 0.4961 (4-mer baseline: ARI 0.1287, NMI 0.1965).
Genomic elements, Block 1: ARI 0.3052, NMI 0.3602.

Collected metrics: [`results/clustering/`](../../results/clustering/).

## Resources

One GPU for feature extraction; UMAP and K-means run on CPU. Feature
extraction over 100,000 sequences × 6 blocks is the dominant cost.
