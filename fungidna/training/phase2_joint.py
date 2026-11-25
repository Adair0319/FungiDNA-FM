"""Phase 2: Joint MLM + Contrastive pretraining on BPE-tokenized fungal genomes.

Single backbone forward — masked input through backbone once,
hidden states feed both MLM head and CL projection head.
With early-stop based on CL loss plateau detection.
"""
import os, sys, math, yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# ── Warmup FlashAttention BEFORE any mamba_ssm import ──
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

from fungidna.model.config import FungiDNAConfig
from fungidna.model.fungi_dna import FungiDNAForJointTraining
from fungidna.data.tokenizer import DualTokenizer
from fungidna.data.pretrain_dataset import Phase2JointDataset, NKBatchSampler, phase2_collate_fn
from fungidna.training.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    print_rank0, get_model, log_grad_norm,
)


class LambdaScheduler:
    """Linear warmup lambda(t) scheduler for CL loss weight."""

    def __init__(self, warmup_steps: int, lambda_max: float = 0.5):
        self.warmup_steps = warmup_steps
        self.lambda_max = lambda_max
        self.current_step = 0

    def get_lambda(self, step: int = None) -> float:
        if step is not None:
            self.current_step = step
        if self.current_step >= self.warmup_steps:
            return self.lambda_max
        return self.lambda_max * self.current_step / max(1, self.warmup_steps)

    def step(self):
        self.current_step += 1


def train_phase2_joint(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    print_rank0(f"Phase 2 Joint Training: {world_size} GPUs")

    # ── Model ──
    model_cfg = FungiDNAConfig()
    model = FungiDNAForJointTraining(model_cfg, num_species=cfg.get("num_species", 735))
    model = model.to(device).bfloat16()

    # Load backbone weights
    resume_backbone = cfg.get("resume_backbone")
    if resume_backbone:
        print_rank0(f"Resuming from Phase 2 backbone: {resume_backbone}")
        backbone_ckpt = torch.load(resume_backbone, map_location=device)
        model.backbone.load_state_dict(backbone_ckpt, strict=True)
        print_rank0("Backbone weights loaded (optimizer/scheduler fresh)")
    elif cfg.get("phase1_checkpoint"):
        print_rank0(f"Loading Phase 1 weights from {cfg['phase1_checkpoint']}")
        ckpt = torch.load(cfg["phase1_checkpoint"], map_location=device)
        model_state = model.state_dict()
        # Load backbone and lm_head (tied weights) from Phase 1 checkpoint
        for k, v in ckpt["model"].items():
            if k in model_state and "projection" not in k:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        print_rank0("Phase 1 weights loaded (projection head randomly initialized)")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    print_rank0(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # ── Tokenizer ──
    tokenizer = DualTokenizer(cfg["tokenizer_path"])

    # ── Dataset ──
    dataset = Phase2JointDataset(
        filtered_fasta_dir=cfg["filtered_fasta_dir"],
        tokenizer=tokenizer,
        window_size=cfg.get("window_size", 8192),
        stride=cfg.get("stride", 8192),
        max_seq_len=cfg.get("max_seq_len", 4096),
        seed=cfg.get("seed", 42) + rank,
    )

    sampler = NKBatchSampler(
        species_to_indices=dataset.species_to_indices,
        n_species=cfg.get("n_species", 8),
        k_per_species=cfg.get("k_per_species", 4),
        seed=cfg.get("seed", 42) + rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=phase2_collate_fn,
    )
    print_rank0(f"Dataset ready: {len(dataset)} windows, {dataset.num_species} species, "
                f"N={cfg.get('n_species', 8)}, K={cfg.get('k_per_species', 4)}")

    # ── Snapshot Matrix (optional) ──
    use_weighted_cl = cfg.get("use_weighted_cl", False)
    W_global = None
    if use_weighted_cl and cfg.get("snapshot_matrix_path"):
        W_global = torch.load(cfg["snapshot_matrix_path"], map_location="cpu")
        print_rank0(f"Snapshot matrix loaded: {W_global.shape}")

    # ── Optimizer ──
    backbone_params = []
    proj_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "projection" in name:
            proj_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.get("lr_backbone", 1.0e-4)},
        {"params": proj_params, "lr": cfg.get("lr_projection", 5.0e-4)},
    ], betas=(0.9, 0.98), weight_decay=0.1)
    print_rank0(f"Optimizer: backbone_lr={cfg.get('lr_backbone', 1.0e-4)}, "
                f"proj_lr={cfg.get('lr_projection', 5.0e-4)}")

    # ── Schedulers ──
    total_steps = cfg["total_steps"]
    warmup_steps = cfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # lambda scheduler (linear warmup, same length as lr warmup)
    lambda_scheduler = LambdaScheduler(
        warmup_steps=warmup_steps,
        lambda_max=cfg.get("lambda_max", 0.5),
    )

    # ── Resume from step (advances schedulers, does NOT recover AdamW momentum) ──
    resume_step = cfg.get("resume_step", 0)
    if resume_step > 0:
        print_rank0(f"Resuming from step {resume_step} (lr={lr_lambda(resume_step)*cfg.get('lr_backbone',1e-4):.2e}, "
                    f"lambda={lambda_scheduler.get_lambda(resume_step):.3f})")
        global_step = resume_step
        # Advance LR scheduler by setting last_epoch
        scheduler.last_epoch = resume_step
        # Advance lambda scheduler
        lambda_scheduler.current_step = resume_step
        # Advance optimizer step counter (affects nothing, just for logging)
        optimizer.zero_grad()
    else:
        global_step = 0

    # ── Wandb ──
    if is_main_process() and cfg.get("wandb_project"):
        wandb.init(project=cfg["wandb_project"], name=cfg.get("wandb_run", "phase2-joint"),
                   config=cfg)

    # ── Early-stop config ──
    eval_every = cfg.get("eval_every", 2000)
    min_steps = cfg.get("min_steps", warmup_steps)  # must finish λ warmup
    early_stop_patience = cfg.get("early_stop_patience", 3)
    early_stop_min_delta = cfg.get("early_stop_min_delta", 0.005)

    # ── Training ──
    model.train()
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    log_every = cfg.get("log_every", 25)
    save_every = cfg.get("save_every", 5000)
    effective_bs = cfg["n_species"] * cfg["k_per_species"] * world_size * grad_accum

    # Early-stop state
    best_cl_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    stop_step = total_steps

    print_rank0(f"Starting training: {total_steps} steps, warmup={warmup_steps}, "
                f"grad_accum={grad_accum}, effective_bs={effective_bs}, "
                f"use_weighted_cl={use_weighted_cl}")
    print_rank0(f"Early-stop: min_steps={min_steps}, eval_every={eval_every}, "
                f"patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    optimizer.zero_grad()

    while global_step < total_steps:
        total_loss_accum = 0.0
        total_mlm_accum = 0.0
        total_cl_accum = 0.0
        micro_step = 0
        n_log_steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            species_labels = batch["species_labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            lambda_val = lambda_scheduler.get_lambda(global_step)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                m = get_model(model) if hasattr(model, "module") else model
                if use_weighted_cl and W_global is not None:
                    batch_W = W_global[species_labels][:, species_labels].to(device)
                    out = m.forward_weighted(
                        input_ids, mlm_labels, species_labels,
                        attention_mask, lambda_val, batch_W,
                    )
                else:
                    out = m(input_ids, mlm_labels, species_labels,
                            attention_mask, lambda_val)
                loss = out["loss"] / grad_accum

            loss.backward()
            micro_step += 1
            total_loss_accum += out["loss"].item()
            total_mlm_accum += out["mlm_loss"].item()
            total_cl_accum += out["cl_loss"].item()
            n_log_steps += 1

            if micro_step % grad_accum == 0:
                grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                lambda_scheduler.step()
                global_step += 1

                if global_step % log_every == 0 and is_main_process():
                    n = max(n_log_steps, 1)
                    avg_loss = total_loss_accum / n
                    avg_mlm = total_mlm_accum / n
                    avg_cl = total_cl_accum / n
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | loss={avg_loss:.4f} "
                          f"(mlm={avg_mlm:.4f} cl={avg_cl:.4f}) "
                          f"lambda={lambda_val:.3f} | grad={grad_norm_val:.2f} | lr={lr:.2e}")
                    if cfg.get("wandb_project"):
                        wandb.log({
                            "loss": avg_loss, "mlm_loss": avg_mlm, "cl_loss": avg_cl,
                            "lambda": lambda_val, "grad_norm": grad_norm_val.item(),
                            "lr": lr, "step": global_step,
                        })
                # ── Evaluation + Early-stop check (BEFORE accumulator reset) ──
                if (global_step >= min_steps
                        and global_step % eval_every == 0
                        and is_main_process()):
                    # Use CL loss from the just-logged window
                    n = max(n_log_steps, 1)
                    eval_cl = total_cl_accum / n

                    if eval_cl > 0:
                        # Handle first eval (best_cl_loss is inf) — always accept
                        is_first_eval = (best_cl_loss == float("inf"))
                        if is_first_eval:
                            improvement = 1.0  # dummy: always treated as improvement
                        else:
                            improvement = (best_cl_loss - eval_cl) / best_cl_loss

                        if is_first_eval or improvement > early_stop_min_delta:
                            best_cl_loss = eval_cl
                            patience_counter = 0
                            print_rank0(f"  Eval @ step {global_step}: cl_loss={eval_cl:.4f} "
                                        f"(improved {improvement*100:.1f}%) — reset patience")
                        else:
                            patience_counter += 1
                            print_rank0(f"  Eval @ step {global_step}: cl_loss={eval_cl:.4f} "
                                        f"(no improvement, patience {patience_counter}/"
                                        f"{early_stop_patience})")

                        if patience_counter >= early_stop_patience:
                            print_rank0(f"Early stop triggered at step {global_step} "
                                        f"(CL loss plateaued at ~{eval_cl:.4f})")
                            stopped_early = True
                            stop_step = global_step
                            break

                    total_loss_accum = 0.0
                    total_mlm_accum = 0.0
                    total_cl_accum = 0.0
                    n_log_steps = 0
                elif global_step % log_every == 0 and is_main_process():
                    # Regular reset (no eval at this step)
                    total_loss_accum = 0.0
                    total_mlm_accum = 0.0
                    total_cl_accum = 0.0
                    n_log_steps = 0

                if global_step % save_every == 0:
                    m_save = get_model(model) if hasattr(model, "module") else model
                    torch.save(
                        m_save.backbone.state_dict(),
                        os.path.join(cfg["output_dir"], f"backbone-{global_step}.pt"),
                    )
                    print_rank0(f"Checkpoint saved at step {global_step}")

                if global_step >= total_steps:
                    break

        # Broadcast early-stop from rank 0
        if world_size > 1:
            stop_tensor = torch.tensor([1 if stopped_early else 0], device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

        if stopped_early:
            break

    # ── Final ──
    m_final = get_model(model) if hasattr(model, "module") else model
    final_path = os.path.join(cfg["output_dir"], "backbone_final.pt")
    torch.save(m_final.backbone.state_dict(), final_path)
    if is_main_process():
        print(f"Phase 2 Joint Training finished at step {stop_step}/{total_steps} "
              f"({'early stop' if stopped_early else 'max steps'})")
        if cfg.get("wandb_project"):
            wandb.log({"final_step": stop_step, "early_stop": int(stopped_early)})
            wandb.finish()
    print_rank0(f"Backbone saved to {final_path}")
    cleanup_distributed()
