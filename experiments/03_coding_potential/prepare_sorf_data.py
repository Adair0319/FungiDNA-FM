"""Prepare human sORF dataset: 641 coding + 641 ncRNA → parquet."""
import os, re, pandas as pd, numpy as np

SEED = 42
np.random.seed(SEED)

DATA_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task_sorf"
OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/cds_intergenic_sorf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_fasta_as_records(path, label):
    records = []
    header, seq_lines = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    seq = ''.join(seq_lines)
                    seqid = re.split(r'[\s|]', header)[0] if header else 'unknown'
                    records.append({'sequence': seq.upper(), 'label': label,
                                    'species': 'human', 'seqid': seqid,
                                    'length': len(seq)})
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            seq = ''.join(seq_lines)
            seqid = re.split(r'[\s|]', header)[0] if header else 'unknown'
            records.append({'sequence': seq.upper(), 'label': label,
                            'species': 'human', 'seqid': seqid, 'length': len(seq)})
    return records

coding = parse_fasta_as_records(os.path.join(DATA_DIR, "Human.small_coding_RNA_test.fa"), 1)
ncrna  = parse_fasta_as_records(os.path.join(DATA_DIR, "Homo38.small_ncrna_test.fa"), 0)
print(f"Coding: {len(coding)}, ncRNA: {len(ncrna)}")

df = pd.DataFrame(coding + ncrna)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split: 80% train+val, 20% test
test_df = df.sample(frac=0.2, random_state=SEED)
trainval_df = df.drop(test_df.index)
val_df = trainval_df.sample(frac=0.1, random_state=SEED)
train_df = trainval_df.drop(val_df.index)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Create fold column: train=1, val=1, test=0
df['fold'] = 0
df.loc[train_df.index, 'fold'] = 1
df.loc[val_df.index, 'fold'] = 1

# Save full dataset + split files
df.to_parquet(os.path.join(OUTPUT_DIR, "dataset.parquet"), index=False)
train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)

print(f"Saved to {OUTPUT_DIR}/")
print(f"  Length: min={df.length.min()}, median={df.length.median():.0f}, max={df.length.max()}")
