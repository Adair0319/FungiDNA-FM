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

#: The two released tokenizer variants. Each has its own backbone checkpoint.
VARIANTS = ("unigram", "snt")


def normalize_variant(variant):
    """Return a validated variant name, raising ``WeightsNotFoundError`` otherwise."""
    name = (variant or "unigram").strip().lower()
    if name not in VARIANTS:
        raise WeightsNotFoundError(
            name, f"unknown variant {variant!r}; supported: {', '.join(VARIANTS)}"
        )
    return name


class WeightsNotFoundError(Exception):
    def __init__(self, task, message=None):
        super().__init__(message or f"no checkpoint registered for task {task!r}")
        self.task = task


class WeightLoader:
    def __init__(self, registry=None):
        self.registry = dict(registry or {})

    def resolve(self, task, rank=None, variant="unigram"):
        """Resolve the checkpoint path for ``task`` (registry → env → YAML).

        ``rank``, when given, resolves a per-rank key ``f"{task}_{rank}"``
        (used by the taxonomy task's per-rank MLP heads). ``variant`` selects the
        ``unigram``/``snt`` section of the weights config, so the two variants'
        checkpoints can never be resolved for one another.
        """
        key = task if rank is None else f"{task}_{rank}"
        return self._resolve_key(key, normalize_variant(variant), allow_flat=True)

    def resolve_backbone(self, variant="unigram"):
        """Resolve the pretrained backbone for ``variant``.

        Unlike task keys, a flat ````backbone:`` entry is *not* accepted: the
        weights config predates variant support with a single backbone, and
        silently reusing it for ``variant="snt"`` would load the Unigram weights.
        """
        return self._resolve_key(
            BACKBONE_KEY, normalize_variant(variant), allow_flat=False
        )

    def resolve_tokenizer(self):
        """Resolve the shared SentencePiece ``.model`` path (registry → env → YAML).

        The tokenizer file is shared by both variants; single-nucleotide encoding
        is algorithmic and needs no file.
        """
        return self._resolve_key(TOKENIZER_KEY, variant=None, allow_flat=True)

    def _resolve_key(self, key, variant, allow_flat):
        if key in self.registry:
            return Path(self.registry[key])
        if variant is not None:
            env_var = f"FUNGIDNA_{variant.upper()}_{key.upper()}_WEIGHTS"
            env_val = os.environ.get(env_var)
            if env_val:
                return Path(env_val)
        if allow_flat:
            env_val = os.environ.get(f"FUNGIDNA_{key.upper()}_WEIGHTS")
            if env_val:
                return Path(env_val)
        path_from_config = self._resolve_from_config_file(key, variant, allow_flat)
        if path_from_config is not None:
            return Path(path_from_config)
        where = f"{key!r} for variant {variant!r}" if variant else repr(key)
        hint = (
            f"; expected a per-variant entry such as ``{variant}.{key}``"
            if variant and not allow_flat
            else ""
        )
        raise WeightsNotFoundError(
            key, f"no checkpoint registered for {where}{hint}"
        )

    def _resolve_from_config_file(self, key, variant, allow_flat):
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
        if not isinstance(data, dict):
            return None
        if variant is not None:
            section = data.get(variant)
            if isinstance(section, dict) and section.get(key):
                return section[key]
        if allow_flat:
            path = data.get(key)
            if path:
                return path
        return None


def _checkpoint_dtype(state_dict):
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            return value.dtype
    return torch.float32


def default_device():
    """The inference device: CUDA when available (required by mamba/flash-attn)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, ckpt_path, unwrap="model", device=None, skip_size_mismatch=()):
    """Load a checkpoint into ``model`` with explicit shape checking.

    ``unwrap`` names a top-level dict key to extract (e.g. ``"model"``); pass
    ``unwrap=None`` for a bare state dict. The model is moved to ``device``
    (default: CUDA if available) and cast to the checkpoint dtype.

    Shape-mismatch policy (never silently skipped):

    * keys whose shape matches are loaded;
    * keys whose shape mismatches AND whose name contains one of the
      ``skip_size_mismatch`` substrings are skipped, with a printed explanation
      (used for the Unigram variant's unused single-nucleotide embedding, which
      is ``(5, 768)`` in the Unigram checkpoint but ``(10, 768)`` in the shared
      architecture — the Unigram variant never reads ``nt_embed`` because it uses
      ``token_type=0`` / the 4096-vocab Unigram embedding);
    * any *other* shape mismatch raises :class:`WeightsNotFoundError`.

    Returns ``(loaded, skipped, missing, unexpected)``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise WeightsNotFoundError(
            str(ckpt_path), f"checkpoint not found: {ckpt_path}"
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if unwrap is not None and isinstance(state, dict) and unwrap in state and isinstance(state[unwrap], dict):
        state = state[unwrap]
    if not isinstance(state, dict) or not any(isinstance(v, torch.Tensor) for v in state.values()):
        raise WeightsNotFoundError(
            str(ckpt_path), f"checkpoint has no tensor state dict: {ckpt_path}"
        )

    dtype = _checkpoint_dtype(state)
    device = device or default_device()
    model = model.to(dtype=dtype).to(device)

    model_sd = model.state_dict()
    filtered = {}
    loaded = 0
    skipped = []
    for key, val in state.items():
        if key not in model_sd:
            continue  # reported as ``unexpected`` below
        if model_sd[key].shape == val.shape:
            filtered[key] = val
            loaded += 1
        elif any(sub in key for sub in skip_size_mismatch):
            skipped.append((key, tuple(val.shape), tuple(model_sd[key].shape)))
        else:
            raise WeightsNotFoundError(
                str(ckpt_path),
                f"shape mismatch for {key!r}: checkpoint {tuple(val.shape)} "
                f"vs model {tuple(model_sd[key].shape)}",
            )

    if loaded == 0:
        raise WeightsNotFoundError(
            str(ckpt_path),
            f"no checkpoint keys match the model for {ckpt_path}; "
            f"checkpoint has {len(state)} keys, model expects {len(model_sd)}.",
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(
            f"[load_checkpoint] {ckpt_path.name}: skipped {len(skipped)} "
            f"size-mismatched key(s) (explicitly allowed): {skipped}"
        )
    print(
        f"[load_checkpoint] {ckpt_path.name}: loaded {loaded}/{len(model_sd)} "
        f"model params; missing={len(missing)} unexpected={len(unexpected)}"
    )
    return loaded, skipped, missing, unexpected


class TaskHandler:
    task = None  # Task enum member, set by subclass

    def run(self, request):
        raise NotImplementedError
