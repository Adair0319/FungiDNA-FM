"""Task-handler base class, pluggable weight loader, and shared load helpers.

The loading helpers centralise the checkpoint conventions that differ across
tasks, so a layout mismatch fails loudly instead of silently leaving a model
randomly initialised:

  * checkpoints saved as ``torch.save({"model": state_dict, ...})`` are unwrapped;
  * keys with a ``backbone.`` / ``head.`` prefix are left as-is (they match the
    ``FungiDNAFor*`` submodules of the same name) and keys without a prefix are
    loaded directly (bare backbone / bare MLP heads);
  * after loading, the number of matched keys is asserted to be > 0;
  * the model is cast to the checkpoint's dtype (bf16 vs fp32 differ per task).
"""
import os
from pathlib import Path

import torch
import yaml

#: Default location of the task -> checkpoint YAML mapping. Overridable via the
#: ``FUNGIDNA_WEIGHTS_CONFIG`` environment variable.
DEFAULT_WEIGHTS_CONFIG = "~/.config/fungidna/weights.yaml"

#: Well-known artifact keys resolved by :meth:`WeightLoader.resolve`.
BACKBONE_KEY = "backbone"
TOKENIZER_KEY = "tokenizer"


class WeightsNotFoundError(Exception):
    def __init__(self, task, message=None):
        super().__init__(message or f"no checkpoint registered for task {task!r}")
        self.task = task


class WeightLoader:
    def __init__(self, registry=None):
        self.registry = dict(registry or {})

    def resolve(self, task):
        """Resolve the checkpoint path for ``task`` (registry → env → YAML)."""
        return self._resolve_key(task)

    def resolve_backbone(self):
        """Resolve the shared pretrained backbone (registry → env → YAML)."""
        return self._resolve_key(BACKBONE_KEY)

    def resolve_tokenizer(self):
        """Resolve the SentencePiece ``.model`` path (registry → env → YAML)."""
        return self._resolve_key(TOKENIZER_KEY)

    def _resolve_key(self, key):
        if key in self.registry:
            return Path(self.registry[key])
        env_var = f"FUNGIDNA_{key.upper()}_WEIGHTS"
        env_val = os.environ.get(env_var)
        if env_val:
            return Path(env_val)
        path_from_config = self._resolve_from_config_file(key)
        if path_from_config is not None:
            return Path(path_from_config)
        raise WeightsNotFoundError(key)

    def _resolve_from_config_file(self, key):
        config_file = Path(
            os.environ.get("FUNGIDNA_WEIGHTS_CONFIG", DEFAULT_WEIGHTS_CONFIG)
        ).expanduser()
        if not config_file.is_file():
            return None
        try:
            with config_file.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            raise WeightsNotFoundError(
                key,
                f"weights config file {str(config_file)} is unreadable or not "
                f"valid YAML: {exc}; no checkpoint registered for {key!r}",
            ) from exc
        if isinstance(data, dict):
            path = data.get(key)
            if path:
                return path
        return None


def _checkpoint_dtype(state_dict):
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            return value.dtype
    return torch.float32


def load_checkpoint(model, ckpt_path, unwrap="model"):
    """Load a checkpoint into ``model`` with loud failure on key mismatch.

    ``unwrap`` names a top-level dict key to extract (e.g. ``"model"``); pass
    ``unwrap=None`` for a bare state dict. Returns ``(matched, missing,
    unexpected)``. Raises ``WeightsNotFoundError`` when the checkpoint is
    missing, cannot be unwrapped, or no keys match the model.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise WeightsNotFoundError(
            str(ckpt_path), f"checkpoint not found: {ckpt_path}"
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if unwrap is not None and isinstance(state, dict) and unwrap in state and isinstance(state[unwrap], dict):
        state = state[unwrap]

    dtype = _checkpoint_dtype(state)
    model = model.to(dtype=dtype)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    matched = model_keys & ckpt_keys
    if not matched:
        raise WeightsNotFoundError(
            str(ckpt_path),
            f"no checkpoint keys match the model for {ckpt_path}; "
            f"checkpoint has {len(ckpt_keys)} keys, model expects {len(model_keys)}. "
            f"Sample ckpt keys: {sorted(ckpt_keys)[:5]}",
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(matched), missing, unexpected


class TaskHandler:
    task = None  # Task enum member, set by subclass

    def run(self, request):
        raise NotImplementedError
