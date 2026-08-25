"""BPE tokenizer training via SentencePiece + DualTokenizer (BPE + single-nt)."""
import os
import random
from pathlib import Path
import sentencepiece as spm
import torch

# Special token IDs — must match SentencePiece training order
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[MASK]", "[UNK]", "[SEP]"]
PAD_ID = 0
CLS_ID = 1
MASK_ID = 2
UNK_ID = 3
SEP_ID = 4

# Single-nucleotide mapping
NT_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
ID_TO_NT = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def train_bpe(
    fasta_dir: str,
    output_model_path: str,
    vocab_size: int = 4096,
    sample_fraction: float = 0.05,
    model_type: str = "unigram",
    max_lines: int = 500_000,
):
    """Train SentencePiece BPE on a random sample of filtered FASTA files.

    Args:
        fasta_dir: Directory containing *.fasta files (filtered output from qc_filter).
        output_model_path: Path for the .model file (will also write .vocab).
        vocab_size: BPE vocabulary size.
        sample_fraction: Fraction of lines to sample from each FASTA.
        model_type: 'unigram' or 'bpe'. Unigram is recommended for DNA.
    """
    # Collect sequences
    texts = []
    fasta_files = list(Path(fasta_dir).glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files found in {fasta_dir}")

    for fp in fasta_files:
        with open(fp) as f:
            lines = [line.strip() for line in f if not line.startswith(">") and line.strip()]
        n_sample = max(1, int(len(lines) * sample_fraction))
        texts.extend(random.sample(lines, min(n_sample, len(lines))))

    # Cap total lines for BPE training speed
    if len(texts) > max_lines:
        texts = random.sample(texts, max_lines)

    # Write to temp file for SentencePiece
    temp_path = output_model_path + ".tmp.txt"
    with open(temp_path, "w") as f:
        f.write("\n".join(texts))

    prefix = output_model_path.replace(".model", "")
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        split_digits=False,
        user_defined_symbols=SPECIAL_TOKENS,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=-1,
        eos_id=-1,
    )
    os.unlink(temp_path)
    print(f"BPE model saved to {output_model_path} (vocab={vocab_size})")


class DualTokenizer:
    """Dual-tokenizer for BPE (short-range tasks) and single-nucleotide (long-range).

    Usage:
        tok = DualTokenizer("data/bpe_fungi.model")
        bpe_ids = tok.encode_bpe("ATCGATCG", add_cls=True)     # → [1, ...]
        nt_ids  = tok.encode_single_nt("ATCGATCG")             # → [0, 3, 1, 2, ...]
    """

    def __init__(self, bpe_model_path: str):
        self.bpe = spm.SentencePieceProcessor()
        self.bpe.load(bpe_model_path)
        self.bpe_vocab_size = self.bpe.vocab_size()

    def encode_bpe(self, sequence: str, add_cls: bool = True) -> list[int]:
        """BPE-encode a DNA sequence. Returns token id list."""
        ids = self.bpe.encode(sequence.upper())
        if add_cls:
            return [CLS_ID] + ids
        return ids

    def encode_single_nt(self, sequence: str, add_cls: bool = False) -> list[int]:
        """Single-nucleotide encode. Returns token id list."""
        ids = [NT_TO_ID.get(c, 4) for c in sequence.upper()]
        if add_cls:
            return [CLS_ID] + ids
        return ids

    def decode_bpe(self, ids: list[int]) -> str:
        return self.bpe.decode(ids)

    @staticmethod
    def decode_single_nt(ids: list[int]) -> str:
        return "".join(ID_TO_NT.get(i, "N") for i in ids)

    @staticmethod
    def single_nt_to_tensor(sequence: str) -> torch.Tensor:
        """Convert DNA string directly to long tensor (single-nt encoding)."""
        return torch.tensor([NT_TO_ID.get(c, 4) for c in sequence.upper()], dtype=torch.long)
