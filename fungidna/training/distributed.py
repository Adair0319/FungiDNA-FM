"""Distributed training utilities for FungiDNA."""
import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize NCCL distributed training. Returns (rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def wait_for_everyone():
    if dist.is_initialized():
        dist.barrier()


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def get_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP wrapper to get the raw model."""
    if hasattr(model, "module"):
        return model.module
    return model


def save_checkpoint(model, optimizer, scheduler, scaler, step: int, path: str, rank: int):
    """Save checkpoint on rank 0 only."""
    if rank != 0:
        return
    model_to_save = get_model(model)
    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")


def log_grad_norm(model: torch.nn.Module, step: int, wandb_run=None) -> float:
    """Compute total gradient norm (L2) across all parameters. Log if wandb_run provided."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if wandb_run is not None:
        wandb_run.log({"grad_norm": total_norm, "step": step})
    return total_norm


def should_evaluate(global_step: int, min_steps: int, eval_every: int,
                    patience: int) -> bool:
    """Whether ``global_step`` is an early-stop evaluation node.

    A non-positive ``patience`` (or ``eval_every``) disables early stopping, so
    no evaluation node is created and no cross-rank synchronisation happens.
    """
    if patience <= 0 or eval_every <= 0:
        return False
    return global_step >= min_steps and global_step % eval_every == 0


def sync_stop_decision(stopped_early: bool, stop_step: int, global_step: int,
                       device) -> tuple[bool, int]:
    """Broadcast rank 0's early-stop decision to every rank.

    Must be called by *all* ranks at every evaluation node, before any rank
    breaks out of the training loop. Returns ``(stopped_early, stop_step)`` with
    identical values on every rank; ``stop_step`` is set to the shared
    ``global_step`` when stopping. In non-distributed runs the decision passes
    through unchanged.
    """
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor([1 if stopped_early else 0], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        stopped_early = bool(flag.item())
        if stopped_early:
            stop_step = global_step
    return stopped_early, stop_step
