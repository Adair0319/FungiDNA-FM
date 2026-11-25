"""Phase 1: Masked Language Modeling pretraining on BPE-tokenized fungal genomes."""
import os, sys, math, yaml
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
import wandb

# ── Warmup FlashAttention BEFORE any mamba_ssm import ──
# mamba_ssm 2.3.x TileLang init corrupts CUDA context, preventing
# FlashAttention's first-time JIT kernel compilation. Warming up
# FlashAttention first avoids this.
def _warmup_flash_attn():
    try:
        from flash_attn import flash_attn_func
        _dev = torch.device("cuda:0")
        _q = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        _k = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        _v = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        flash_attn_func(_q, _k, _v, causal=False)
    except Exception:
        pass

_warmup_flash_attn()
# ──────────────────────────────────────────────────────

from fungidna.model.config import FungiDNAConfig
from fungidna.model.fungi_dna import FungiDNAForMLM
from fungidna.data.tokenizer import DualTokenizer
from fungidna.data.pretrain_dataset import Phase1MLMDataset
from fungidna.training.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    save_checkpoint, print_rank0, should_evaluate, sync_stop_decision,
)


def train_phase1(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    print_rank0(f"Phase 1 MLM: {world_size} GPUs, device={device}")

    # ── Variant ──
    variant = str(cfg.get("variant", "unigram")).lower()
    if variant not in ("unigram", "snt"):
        raise ValueError(f"unknown variant {variant!r}; expected 'unigram' or 'snt'")
    token_type = 1 if variant == "snt" else 0
    print_rank0(f"Variant: {variant} (token_type={token_type})")

    # ── Model ──
    model_cfg = FungiDNAConfig()
    model = FungiDNAForMLM(model_cfg, token_type=token_type).to(device).bfloat16()
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    print_rank0(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # ── Tokenizer ──
    tokenizer = DualTokenizer(cfg["tokenizer_path"])
    print_rank0(f"Tokenizer loaded: vocab={tokenizer.bpe_vocab_size}")

    # ── Dataset ──
    dataset = Phase1MLMDataset(
        filtered_fasta_dir=cfg["filtered_fasta_dir"],
        tokenizer=tokenizer,
        window_sizes=cfg.get("window_sizes", [512, 1024, 2048]),
        stride_fraction=cfg.get("stride_fraction", 0.25),
        mask_rate=cfg.get("mask_rate", 0.15),
        span_mask_fraction=cfg.get("span_mask_fraction", 0.30),
        seed=cfg.get("seed", 42) + rank,
        token_type=token_type,
    )

    def pad_collate_fn(batch):
        input_ids = [item["input_ids"] if isinstance(item["input_ids"], torch.Tensor)
                     else torch.tensor(item["input_ids"]) for item in batch]
        labels = [item["labels"] if isinstance(item["labels"], torch.Tensor)
                  else torch.tensor(item["labels"]) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # attention_mask: 1 for real tokens, 0 for padding
        attention_mask = (padded_input_ids != 0).to(dtype=torch.long)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }
    # =======================================

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size_per_gpu"],
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    print_rank0(f"Dataset ready: {len(os.listdir(cfg['filtered_fasta_dir']))} genomes")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )
    total_steps = cfg["total_steps"]
    warmup_steps = cfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # bfloat16 has same exponent range as float32 — GradScaler not needed

    # ── Resume ──
    start_step = 0
    if cfg.get("resume_from") and os.path.exists(cfg["resume_from"]):
        ckpt = torch.load(cfg["resume_from"], map_location=device)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print_rank0(f"Resumed from step {start_step}")

    # ── Wandb ──
    if is_main_process() and cfg.get("wandb_project"):
        wandb.init(project=cfg["wandb_project"], name=cfg.get("wandb_run", "phase1-mlm"), config=cfg)

    # ── Training Loop ──
    model.train()
    global_step = start_step
    log_every = cfg.get("log_every", 50)
    save_every = cfg.get("save_every", 5000)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    effective_bs = cfg["batch_size_per_gpu"] * world_size * grad_accum

    # Early stop on the MLM-loss plateau. Note S2 reports that both variants
    # entered Phase 2 once the loss had converged (step 10,000 for SNT and
    # 60,000 for Unigram), which requires this check to exist.
    eval_every = cfg.get("eval_every", 2000)
    min_steps = cfg.get("min_steps", warmup_steps)
    early_stop_patience = cfg.get("early_stop_patience", 5)
    early_stop_min_delta = cfg.get("early_stop_min_delta", 0.005)
    best_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    stop_step = total_steps

    print_rank0(f"Starting training: {total_steps} steps, warmup={warmup_steps}, "
                f"grad_accum={grad_accum}, effective_bs={effective_bs}")
    print_rank0(f"Early-stop: min_steps={min_steps}, eval_every={eval_every}, "
                f"patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    optimizer.zero_grad()
    while global_step < total_steps:
        micro_step = 0
        loss_accum = 0.0
        n_log_steps = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels, add_cls=False)
                loss = out["loss"] / grad_accum

            loss.backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                # clip_grad_norm_ returns total norm BEFORE clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                loss_accum += loss.item() * grad_accum
                n_log_steps += 1
                global_step += 1

                if global_step % log_every == 0 and is_main_process():
                    n = max(n_log_steps, 1)
                    avg_loss = loss_accum / n
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | loss={avg_loss:.4f} | "
                          f"grad_norm={grad_norm:.2f} | lr={lr:.2e}")
                    if cfg.get("wandb_project"):
                        wandb.log({"loss": avg_loss, "grad_norm": grad_norm,
                                   "lr": lr, "step": global_step})

                # ── Early-stop check ──
                # rank 0 computes the metric and the decision; every rank then
                # meets at the broadcast below, so no rank leaves the loop early.
                # The metric is the running training loss (no separate validation
                # forward), so no DDP forward collective happens here.
                if should_evaluate(global_step, min_steps, eval_every,
                                   early_stop_patience):
                    if is_main_process():
                        n = max(n_log_steps, 1)
                        eval_loss = loss_accum / n
                        is_first_eval = (best_loss == float("inf"))
                        improvement = 1.0 if is_first_eval else (best_loss - eval_loss) / best_loss

                        if is_first_eval or improvement > early_stop_min_delta:
                            best_loss = eval_loss
                            patience_counter = 0
                            print_rank0(f"  Eval @ step {global_step}: loss={eval_loss:.4f} "
                                        f"(improved {improvement*100:.1f}%) — reset patience")
                        else:
                            patience_counter += 1
                            print_rank0(f"  Eval @ step {global_step}: loss={eval_loss:.4f} "
                                        f"(no improvement, patience {patience_counter}/"
                                        f"{early_stop_patience})")

                        if patience_counter >= early_stop_patience:
                            print_rank0(f"Early stop triggered at step {global_step} "
                                        f"(loss plateaued at ~{eval_loss:.4f})")
                            stopped_early = True
                            stop_step = global_step

                    # All ranks reach this point at every evaluation node.
                    stopped_early, stop_step = sync_stop_decision(
                        stopped_early, stop_step, global_step, device)

                    loss_accum = 0.0
                    n_log_steps = 0
                    if stopped_early:
                        break
                elif global_step % log_every == 0 and is_main_process():
                    loss_accum = 0.0
                    n_log_steps = 0

                if global_step % save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, None, global_step,
                        cfg["output_dir"] + f"/checkpoint-{global_step}.pt", rank,
                    )

                if global_step >= total_steps:
                    break

        if stopped_early:
            break

    # ── Final Save ──
    # ``final_model.pt`` is the state at the moment training ended (max steps or
    # early stop). It is NOT guaranteed to be the lowest-loss checkpoint: there is
    # no separate best-model checkpoint, and the periodic ``checkpoint-<step>.pt``
    # files hold the state at their own step.
    save_checkpoint(
        model, optimizer, scheduler, None, global_step,
        os.path.join(cfg["output_dir"], "final_model.pt"), rank,
    )
    if is_main_process() and cfg.get("wandb_project"):
        wandb.finish()
    print_rank0(f"Phase 1 MLM training complete (step {stop_step}/{total_steps}"
                f"{', early stop' if stopped_early else ''}).")
    cleanup_distributed()


if __name__ == "__main__":
    train_phase1(sys.argv[1])
