"""
CDS vs Intergenic binary classification — unified training script.

Models: ours (FungiDNA), gena (BERT-base-yeast), cnn (CNNBaseline)
Modes: frozen (linear probe), finetune, train (from scratch)
5-fold CV, early stopping, per-sequence predictions CSV.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cds_intergenic.py --model ours --mode frozen
  CUDA_VISIBLE_DEVICES=1 python scripts/train_cds_intergenic.py --model ours --mode finetune
  CUDA_VISIBLE_DEVICES=2 python scripts/train_cds_intergenic.py --model gena  --mode finetune
  CUDA_VISIBLE_DEVICES=6 python scripts/train_cds_intergenic.py --model cnn   --mode train
"""

import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── FlashAttention warmup (deferred to after arg parse) ──
def _warmup_flash_attn(gpu=0):
    try:
        from flash_attn import flash_attn_func
        dev = torch.device(f"cuda:{gpu}")
        q = torch.randn(1, 64, 12, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, 64, 12, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, 64, 12, 64, device=dev, dtype=torch.bfloat16)
        flash_attn_func(q, k, v, causal=False)
    except Exception:
        pass

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    choices=["ours", "gena", "cnn", "mamba2"])
parser.add_argument("--mode", type=str, required=True,
                    choices=["frozen", "finetune", "train"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--train-data", type=str, default=None,
                    help="Override training parquet (few-shot mode)")
parser.add_argument("--test-data", type=str, default=None,
                    help="Override test parquet (few-shot mode)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory")
parser.add_argument("--data-path", type=str, default=None,
                    help="Override dataset parquet for 5-fold CV mode")
args = parser.parse_args()

MODEL_TYPE = args.model
MODE = args.mode

# ── Config ──
EPOCHS = 30

# Model-specific batch sizes (to avoid OOM)
BATCH_MAP = {
    ("ours", "frozen"): 256,
    ("ours", "finetune"): 64,   # 113M params full finetune needs smaller batches
    ("gena", "frozen"): 32,    # BERT 512-token attention
    ("gena", "finetune"): 32,   # BERT 512-token attention is O(n²)
    ("cnn", "train"): 256,
    ("mamba2", "train"): 256,  # single-layer Mamba2, lightweight
}
BATCH_SIZE = 256  # default, overridden below
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 5
SEED = 42

# Gradient accumulation — effective batch = BATCH_SIZE * GRAD_ACCUM
GRAD_ACCUM_MAP = {
    ("ours", "frozen"): 1,
    ("ours", "finetune"): 4,    # 64 * 4 = 256 effective
    ("gena", "frozen"): 8,     # 32 * 8 = 256 effective
    ("gena", "finetune"): 8,    # 32 * 8 = 256 effective
    ("cnn", "train"): 1,
    ("mamba2", "train"): 1,
}

LR_MAP = {
    ("ours", "frozen"): 1e-3,
    ("ours", "finetune"): 5e-5,
    ("gena", "frozen"): 1e-3,  # only train head
    ("gena", "finetune"): 5e-5,
    ("cnn", "train"): 5e-4,
    ("mamba2", "train"): 1e-3,
}

LR = LR_MAP[(MODEL_TYPE, MODE)]
BATCH_SIZE = BATCH_MAP.get((MODEL_TYPE, MODE), 256)
GRAD_ACCUM = GRAD_ACCUM_MAP.get((MODEL_TYPE, MODE), 1)

# Paths
PROJECT_ROOT = "/home/lty/yy_projects/fungi_project/fungi_dna_model"
DATA_PATH = os.path.join(PROJECT_ROOT, "data/downstream/task_cds_intergenic/dataset.parquet")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data/processed/bpe_fungi.model")
OURS_CKPT = os.path.join(PROJECT_ROOT, "checkpoints/phase2_joint/backbone_final.pt")
GENA_MODEL_DIR = os.path.join(PROJECT_ROOT, "baselines/models/gena-lm-bert-base-yeast")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "checkpoints/cds_intergenic")
OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(OUTPUT_BASE, f"{MODEL_TYPE}_{MODE}")

N_FOLDS = 5
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"=== CDS vs Intergenic: {MODEL_TYPE} / {MODE} ===")
print(f"GPU: {args.gpu}, LR: {LR}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print(f"Device: {device}")

_warmup_flash_attn(args.gpu)


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

def tokenize_bpe(text, tokenizer):
    """Encode DNA sequence with BPE tokenizer, add CLS token."""
    ids = tokenizer.encode_bpe(text, add_cls=True)
    return torch.tensor(ids, dtype=torch.long)


class CDSIntergenicDataset(Dataset):
    """Reads from parquet, pre-tokenizes all sequences upfront."""

    def __init__(self, df, tokenizer_fn):
        self.df = df.reset_index(drop=True)
        # Pre-tokenize all sequences upfront (much faster than per-__getitem__)
        self.input_ids = []
        self.labels = []
        self.species = []
        self.seqids = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            self.input_ids.append(tokenizer_fn(row["sequence"]))
            self.labels.append(int(row["label"]))
            self.species.append(row["species"])
            self.seqids.append(row["seqid"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.species[idx], self.seqids[idx]


def collate_fn(batch):
    input_ids, labels, species, seqids = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels, dtype=torch.long), list(species), list(seqids)


# ═══════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════

def build_ours_model(mode):
    from fungidna.model.config import FungiDNAConfig
    from fungidna.model.fungi_dna import FungiDNAForSequenceClassification

    config = FungiDNAConfig()
    model = FungiDNAForSequenceClassification(config, num_classes=2)

    cpt = torch.load(OURS_CKPT, map_location="cpu", weights_only=False)
    bb_state = {}
    for k, v in cpt.items():
        if k.startswith("backbone."):
            bb_state[k[9:]] = v
        else:
            bb_state[k] = v
    missing, unexpected = model.backbone.load_state_dict(bb_state, strict=False)
    print(f"  Backbone: {len(missing)} missing, {len(unexpected)} unexpected")

    if mode == "frozen":
        for p in model.backbone.parameters():
            p.requires_grad = False

    head = nn.Identity()  # model already has built-in ClassificationHead
    return model, head, config.hidden_size


def build_gena_model(mode="finetune"):
    import importlib.util
    # Load custom BertModel from the model directory (has LayerNorm naming fix)
    modeling_path = os.path.join(GENA_MODEL_DIR, "modeling_bert.py")
    spec = importlib.util.spec_from_file_location("gena_bert", modeling_path)
    gena_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gena_module)
    BertModel = gena_module.BertModel
    model = BertModel.from_pretrained(GENA_MODEL_DIR)
    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    head = nn.Linear(model.config.hidden_size, 2)
    return model, head, model.config.hidden_size


def build_mamba2_model():
    """Single-layer Mamba2 with one-hot input (splice baseline architecture)."""
    from mamba_ssm import Mamba2 as Mamba2Layer
    D_MODEL = 256

    class UnifiedHead(nn.Module):
        def __init__(self, input_dim=256, hidden_dim=256, num_classes=2, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            x = self.norm(x)
            x = self.dropout(self.act(self.fc1(x)))
            return self.fc2(x)

    class Mamba2Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(4, D_MODEL)
            self.norm = nn.RMSNorm(D_MODEL)
            self.mamba = Mamba2Layer(d_model=D_MODEL, d_state=128, d_conv=4, expand=2)
            self.head = UnifiedHead(D_MODEL)
        def forward(self, x):
            # x: (B, L, 4) one-hot
            x = self.embed(x)
            x = self.norm(x)
            x = self.mamba(x)
            x = x.mean(dim=1)
            return self.head(x)

    model = Mamba2Baseline()
    head = nn.Identity()
    return model, head, D_MODEL


def build_cnn_model():
    from fungidna.model.baseline_models import CNNBaseline
    model = CNNBaseline(vocab_size=4096, embedding_dim=128, num_classes=2, dropout=0.3)
    head = nn.Identity()  # CNN has built-in classifier
    return model, head, 256


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(labels, probs):
    preds = probs.argmax(axis=-1)
    pos_probs = probs[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, pos_probs)),
        "auprc": float(average_precision_score(labels, pos_probs)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def forward_model(model, head, input_ids, is_frozen=False):
    """Forward pass handling both built-in-head (ours) and separate-head (gena/cnn) models.
    For frozen mode: backbone runs under torch.no_grad() to save activation memory."""
    if is_frozen:
        # Frozen: run backbone under no_grad, then detach before head
        if hasattr(model, 'backbone'):
            # Ours: model has .backbone (StripedMambaBackbone) + .head (ClassificationHead)
            with torch.no_grad():
                features = model.backbone(input_ids)
            if isinstance(features, tuple):
                features = features[0].detach()
            elif hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state.detach()
            else:
                features = features.detach()
            return model.head(features)
        else:
            # GENA/CNN: model IS the backbone, head is separate
            with torch.no_grad():
                features = model(input_ids)
            if isinstance(features, tuple):
                features = features[0].detach()
            elif hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state.detach()
            else:
                features = features.detach()
            if features.ndim == 3:
                features = features.mean(dim=1)
            return head(features)

    output = model(input_ids)

    # If model already returns logits (e.g. FungiDNAForSequenceClassification)
    if isinstance(output, torch.Tensor) and output.ndim == 2 and output.shape[1] == 2:
        return output

    # Extract hidden states from various possible outputs
    if isinstance(output, tuple):
        features = output[0]
    elif hasattr(output, 'last_hidden_state'):
        features = output.last_hidden_state
    else:
        features = output

    # Mean pool over sequence dim
    if features.ndim == 3:
        pooled = features.mean(dim=1)
    else:
        pooled = features

    if isinstance(head, nn.Identity):
        return pooled
    return head(pooled)


@torch.no_grad()
def evaluate(model, head, dataloader, device):
    model.eval()
    if not isinstance(head, nn.Identity):
        head.eval()

    all_labels, all_probs = [], []
    all_species, all_seqids = [], []

    for input_ids, labels, species, seqids in dataloader:
        input_ids = input_ids.to(device)
        labels_np = labels.numpy()
        logits = forward_model(model, head, input_ids, is_frozen=(MODE == "frozen"))
        probs = F.softmax(logits, dim=-1)

        all_labels.append(labels_np)
        all_probs.append(probs.cpu().float().numpy())
        all_species.extend(species)
        all_seqids.extend(seqids)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return compute_metrics(all_labels, all_probs), all_labels, all_probs, all_species, all_seqids


# ═══════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════

def train_one_fold(fold_idx, df, tokenizer_fn):
    os.makedirs(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}"), exist_ok=True)

    # Split
    test_df = df[df["fold"] == fold_idx]
    trainval_df = df[df["fold"] != fold_idx]

    val_size = int(len(trainval_df) * 0.1)
    val_df = trainval_df.sample(n=val_size, random_state=SEED + fold_idx)
    train_df = trainval_df.drop(val_df.index)

    print(f"\n--- Fold {fold_idx} ---")
    print(f"  Train: {len(train_df):,} (pos={train_df.label.sum():,})")
    print(f"  Val:   {len(val_df):,} (pos={val_df.label.sum():,})")
    print(f"  Test:  {len(test_df):,} (pos={test_df.label.sum():,})")

    print(f"  Tokenizing train set ({len(train_df):,} samples)...", end=" ", flush=True)
    t0 = time.time()
    train_ds = CDSIntergenicDataset(train_df, tokenizer_fn)
    print(f"{time.time()-t0:.1f}s", flush=True)
    print(f"  Tokenizing val set ({len(val_df):,} samples)...", end=" ", flush=True)
    t0 = time.time()
    val_ds = CDSIntergenicDataset(val_df, tokenizer_fn)
    print(f"{time.time()-t0:.1f}s", flush=True)
    print(f"  Tokenizing test set ({len(test_df):,} samples)...", end=" ", flush=True)
    t0 = time.time()
    test_ds = CDSIntergenicDataset(test_df, tokenizer_fn)
    print(f"{time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0,
                             pin_memory=True, drop_last=False)

    # Build model
    if MODEL_TYPE == "ours":
        model, head, hidden_size = build_ours_model(MODE)
    elif MODEL_TYPE == "gena":
        model, head, hidden_size = build_gena_model(MODE)
    elif MODEL_TYPE == "mamba2":
        model, head, hidden_size = build_mamba2_model()
    else:
        model, head, hidden_size = build_cnn_model()

    # bf16 for ours/gena/cnn; fp32 for mamba2 (one-hot input)
    if MODEL_TYPE == "mamba2":
        model = model.to(device)
        if not isinstance(head, nn.Identity):
            head = head.to(device)
    else:
        model = model.to(device).bfloat16()
        if not isinstance(head, nn.Identity):
            head = head.to(device).bfloat16()

    # Optimizer — collect all trainable params
    all_params = list(model.parameters())
    if not isinstance(head, nn.Identity):
        all_params += list(head.parameters())
    trainable_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_trainable = sum(p.numel() for p in trainable_params)
    total_all = sum(p.numel() for p in all_params)
    print(f"  Params: {total_trainable:,} trainable / {total_all:,} total ({100*total_trainable/max(total_all,1):.1f}%)")

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * EPOCHS
    print(f"  Batch: {BATCH_SIZE}, GradAccum: {GRAD_ACCUM}, Effective: {BATCH_SIZE*GRAD_ACCUM}")
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(EPOCHS):
        model.train()
        if not isinstance(head, nn.Identity):
            head.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, labels, _, _) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_model(model, head, input_ids, is_frozen=(MODE == "frozen"))
                loss = criterion(logits, labels) / GRAD_ACCUM

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in all_params if p.requires_grad], MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        history["train_loss"].append(avg_loss)

        # Validation
        val_metrics, _, _, _, _ = evaluate(model, head, val_loader, device)
        val_loss_calc = val_metrics["f1_macro"]  # proxy via F1
        history["val_loss"].append(avg_loss)
        history["val_auroc"].append(val_metrics["auroc"])

        # Compute actual val loss for early stopping
        val_loss_val = 0.0
        model.eval()
        if not isinstance(head, nn.Identity):
            head.eval()
        with torch.no_grad():
            for input_ids, labels, _, _ in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = forward_model(model, head, input_ids, is_frozen=(MODE == "frozen"))
                    val_loss_val += criterion(logits, labels).item()
        val_loss_val /= max(len(val_loader), 1)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | loss={avg_loss:.4f} val_loss={val_loss_val:.4f} "
              f"auroc={val_metrics['auroc']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        # Early stopping on val loss
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = epoch + 1
            patience_counter = 0
            ckpt = {
                "model": model.state_dict(),
                "head": {} if isinstance(head, nn.Identity) else head.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    # Load best and evaluate test
    best = torch.load(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "best_model.pt"),
                      map_location=device, weights_only=False)

    if MODEL_TYPE == "ours":
        model, head, _ = build_ours_model(MODE)
    elif MODEL_TYPE == "gena":
        model, head, _ = build_gena_model(MODE)
    elif MODEL_TYPE == "mamba2":
        model, head, _ = build_mamba2_model()
    else:
        model, head, _ = build_cnn_model()
    model.load_state_dict(best["model"], strict=False)
    model = model.to(device)
    if MODEL_TYPE != "mamba2":
        model = model.bfloat16()

    if not isinstance(head, nn.Identity) and best.get("head"):
        head.load_state_dict(best["head"])
    if not isinstance(head, nn.Identity):
        head = head.to(device)
        if MODEL_TYPE != "mamba2":
            head = head.bfloat16()

    test_metrics, test_labels, test_probs, test_species, test_seqids = evaluate(
        model, head, test_loader, device)
    test_preds = test_probs.argmax(axis=-1)

    print(f"  Test: AUROC={test_metrics['auroc']:.4f} AUPRC={test_metrics['auprc']:.4f} "
          f"F1={test_metrics['f1_macro']:.4f} MCC={test_metrics['mcc']:.4f} "
          f"Acc={test_metrics['accuracy']:.4f}")

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "metrics.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, **test_metrics}, f, indent=2)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "fold": fold_idx,
        "species": test_species,
        "seqid": test_seqids,
        "true_label": test_labels.astype(int),
        "pred_label": test_preds.astype(int),
        "prob_cds": test_probs[:, 1],
        "prob_intergenic": test_probs[:, 0],
        "correct": (test_labels == test_preds).astype(int),
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "predictions.csv"), index=False)

    return test_metrics


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")

    # Build tokenizer
    if MODEL_TYPE == "mamba2":
        NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
                   'a': 0, 'c': 1, 'g': 2, 't': 3}
        def tokenize(seq):
            ids = [NUC_MAP.get(c, 4) for c in seq]  # 4 = N/unknown
            onehot = F.one_hot(torch.tensor(ids, dtype=torch.long), num_classes=5)
            onehot = onehot[:, :4]  # keep only ATCG channels; unknown chars → all-zero
            return onehot.float()
    elif MODEL_TYPE == "gena":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(GENA_MODEL_DIR)
        def tokenize(seq):
            tokens = tokenizer(seq, return_tensors="pt", padding=False, truncation=True,
                              max_length=512).input_ids.squeeze(0)
            return tokens
    else:
        from fungidna.data.tokenizer import DualTokenizer
        bpe = DualTokenizer(TOKENIZER_PATH)
        def tokenize(seq):
            return tokenize_bpe(seq, bpe)

    # Few-shot mode: single training run on specified data
    if args.train_data is not None:
        print(f"\n=== FEW-SHOT MODE ===")
        print(f"Loading train data: {args.train_data}")
        train_df = pd.read_parquet(args.train_data)
        print(f"  Train: {len(train_df):,} (pos={train_df.label.sum():,})")

        # Load test set (fold 0)
        if args.test_data:
            test_path = args.test_data
        else:
            test_path = os.path.join(os.path.dirname(args.train_data), "test_fold0.parquet")
        print(f"Loading test data: {test_path}")
        test_df = pd.read_parquet(test_path)
        print(f"  Test: {len(test_df):,} (pos={test_df.label.sum():,})")

        # Build a df with fold=0 for test, fold≠0 for train
        test_df["fold"] = 0
        train_df["fold"] = 1
        df = pd.concat([train_df, test_df], ignore_index=True)

        metrics = train_one_fold(0, df, tokenize)  # fold 0 = test
        print(json.dumps(metrics, indent=2))

        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nDone. Output: {OUTPUT_DIR}/")
        return

    # Regular 5-fold CV mode
    # Load dataset
    data_path = args.data_path if args.data_path else DATA_PATH
    print(f"\nLoading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  Total: {len(df):,} (pos={df.label.sum():,}, neg={(df.label==0).sum():,})")

    all_metrics = []
    for fold_idx in range(N_FOLDS):
        metrics = train_one_fold(fold_idx, df, tokenize)
        all_metrics.append(metrics)

    # CV summary
    keys = ["accuracy", "auroc", "auprc", "f1_macro", "mcc"]
    summary = {"model": f"{MODEL_TYPE}_{MODE}", "n_folds": N_FOLDS, "metrics": {}}
    print(f"\n{'='*60}")
    print(f"CV Summary — {MODEL_TYPE}/{MODE}")
    print(f"{'='*60}")

    for key in keys:
        vals = [m[key] for m in all_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary["metrics"][key] = {"mean": float(mean_val), "std": float(std_val)}
        print(f"  {key:12s}: {mean_val:.4f} ± {std_val:.4f}")

    best_fold = np.argmax([m["auroc"] for m in all_metrics])
    summary["best_fold"] = int(best_fold)
    summary["best_fold_auroc"] = float(all_metrics[best_fold]["auroc"])

    with open(os.path.join(OUTPUT_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Combine all predictions
    all_preds = []
    for fold_idx in range(N_FOLDS):
        csv_path = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}", "predictions.csv")
        if os.path.exists(csv_path):
            all_preds.append(pd.read_csv(csv_path))
    if all_preds:
        pd.concat(all_preds).to_csv(os.path.join(OUTPUT_DIR, "all_predictions.csv"), index=False)

    print(f"\nDone. Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
