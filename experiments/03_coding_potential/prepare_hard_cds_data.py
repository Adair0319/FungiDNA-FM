"""Build Hard CDS/Intergenic dataset: short CDS <= 300bp + length-matched intergenic windows."""
import os, re, random, numpy as np, pandas as pd

SEED = 42; random.seed(SEED); np.random.seed(SEED)

CDS_FASTA = "/home/lty/yy_projects/fungi_project/1kfg_datasets/output/merged/cds.fasta"
INTERGENIC_FASTA = "/home/lty/yy_projects/fungi_project/1kfg_datasets/output/merged/intergenic.fasta"
OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/cds_intergenic_hard"
N_PER_CLASS = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Collect all short CDS
print("Collecting short CDS (<=300bp)...")
short_cds = []
header, seq_lines = None, []
with open(CDS_FASTA) as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if header is not None:
                seq = ''.join(seq_lines)
                if len(seq) <= 300:
                    # Parse header: species|transcript_id|gene_name|...
                    parts = header.split('|')
                    short_cds.append({
                        'header': header,
                        'sequence': seq.upper(),
                        'length': len(seq),
                        'species': parts[0] if len(parts) >= 1 else 'unknown',
                        'seqid': parts[1] if len(parts) >= 2 else 'unknown',
                    })
            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        seq = ''.join(seq_lines)
        if len(seq) <= 300:
            parts = header.split('|')
            short_cds.append({'header': header, 'sequence': seq.upper(),
                              'length': len(seq),
                              'species': parts[0] if len(parts) >= 1 else 'unknown',
                              'seqid': parts[1] if len(parts) >= 2 else 'unknown'})

print(f"  Found {len(short_cds):,} short CDS")

# Step 2: Sample N CDS
sampled_cds = random.sample(short_cds, N_PER_CLASS)
cds_lengths = [r['length'] for r in sampled_cds]
print(f"  Sampled {len(sampled_cds):,}, lengths: min={min(cds_lengths)}, median={np.median(cds_lengths):.0f}, max={max(cds_lengths)}")

# Step 3: Build intergenic index — store positions of long intergenic regions for windowing
print("Building intergenic index...")
intergenic_regions = []  # [(species, seqid, full_seq, len), ...]
header, seq_lines = None, []
with open(INTERGENIC_FASTA) as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if header is not None:
                seq = ''.join(seq_lines)
                if len(seq) > 50:  # need at least some length for windowing
                    parts = header.split('|')
                    intergenic_regions.append({
                        'header': header,
                        'sequence': seq.upper(),
                        'length': len(seq),
                        'species': parts[0] if len(parts) >= 1 else 'unknown',
                        'seqid': parts[1] if len(parts) >= 2 else 'unknown',
                    })
            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        seq = ''.join(seq_lines)
        if len(seq) > 50:
            parts = header.split('|')
            intergenic_regions.append({'header': header, 'sequence': seq.upper(),
                                       'length': len(seq),
                                       'species': parts[0] if len(parts) >= 1 else 'unknown',
                                       'seqid': parts[1] if len(parts) >= 2 else 'unknown'})

print(f"  {len(intergenic_regions):,} intergenic regions available")

# Step 4: For each CDS length, randomly window an intergenic region to match
print(f"Generating {N_PER_CLASS:,} length-matched intergenic samples...")
intergenic_samples = []
for cds_rec in sampled_cds:
    target_len = cds_rec['length']
    # Find a suitable intergenic region
    for _ in range(100):  # retry up to 100 times
        region = random.choice(intergenic_regions)
        if region['length'] >= target_len:
            start = random.randint(0, region['length'] - target_len)
            seq = region['sequence'][start:start + target_len]
            intergenic_samples.append({
                'sequence': seq,
                'length': target_len,
                'species': region['species'],
                'seqid': f"{region['seqid']}:{start}-{start+target_len}",
                'start': start,
            })
            break

    if len(intergenic_samples) % 2000 == 0:
        print(f"  {len(intergenic_samples):,}/{N_PER_CLASS:,}...")

print(f"  Generated {len(intergenic_samples):,}")

# Step 5: Build parquet
positive = [{'sequence': r['sequence'], 'label': 1, 'species': r['species'],
              'seqid': r['seqid'], 'length': r['length']} for r in sampled_cds]
negative = [{'sequence': r['sequence'], 'label': 0, 'species': r['species'],
              'seqid': r['seqid'], 'length': r['length'], 'start': r['start']} for r in intergenic_samples]

df = pd.DataFrame(positive + negative)
df['softmask_ratio'] = 0.0
if 'start' not in df.columns:
    df['start'] = -1
df['start'] = df['start'].fillna(-1).astype(int)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split 80/20, val 10% from train
test_df = df.sample(frac=0.2, random_state=SEED)
trainval_df = df.drop(test_df.index)
val_df = trainval_df.sample(frac=0.1, random_state=SEED)
train_df = trainval_df.drop(val_df.index)

print(f"\nSplit: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
print(f"Length: min={df.length.min()}, median={df.length.median():.0f}, max={df.length.max()}")

# Verify length matching
pos_l = df[df.label==1].length
neg_l = df[df.label==0].length
print(f"CDS lengths: mean={pos_l.mean():.1f}, std={pos_l.std():.1f}")
print(f"Intergenic lengths: mean={neg_l.mean():.1f}, std={neg_l.std():.1f}")

df.to_parquet(os.path.join(OUTPUT_DIR, "dataset.parquet"), index=False)
train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)
print(f"\nSaved to {OUTPUT_DIR}/")
