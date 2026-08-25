#!/bin/bash
cd /home/lty/yy_projects/fungi_project/fungi_dna_model
exec /home/lty/miniconda3/envs/fungi/bin/python -u scripts/umap_clustering_elements.py > analysis/umap_elements/run_output.log 2>&1
