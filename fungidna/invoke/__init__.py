"""FungiDNA guarded invocation package.

Public entry points:

- ``run_task(file_path, task)`` — embedded-API path: ``validate`` + ``dispatch``.
- ``parse_prompt(text, llm_extractor=None)`` — natural-language terminal path.
- ``python -m fungidna`` — CLI wrapper over ``parse_prompt``/``validate``/``dispatch``.

Import-time note: the five concrete handler classes live in
``fungidna.invoke.handlers.*`` modules that do ``import torch`` at module scope,
and they load the mamba/flash-attn model chain lazily inside ``run()``.  To keep
``import fungidna.invoke`` free of the torch/mamba/flash-attn dependency stack
(validation and parse errors must be reachable without a DL runtime), this
package registers *lazy proxies* for the five handlers at import time.  The real
handler module is imported only when ``dispatch()`` first needs it — i.e. after
the deterministic validation gate has passed.
"""
import importlib

from .dispatcher import dispatch, register
from .parser import parse_prompt
from .report import Report
from .request import StructuredRequest, Task, task_from_string
from .validation import ValidationError, validate

# task -> (module path, handler class name).  Kept here so all five handlers are
# registered with the dispatcher at import time without importing their modules
# (and therefore without importing torch).
_HANDLER_FACTORIES = {
    Task.REPRESENTATION: ("fungidna.invoke.handlers.representation", "RepresentationHandler"),
    Task.TAXONOMY: ("fungidna.invoke.handlers.taxonomy", "TaxonomyHandler"),
    Task.CODING_POTENTIAL: ("fungidna.invoke.handlers.coding_potential", "CodingPotentialHandler"),
    Task.SPLICE_SITE: ("fungidna.invoke.handlers.splice_site", "SpliceSiteHandler"),
    Task.BGC_BOUNDARY: ("fungidna.invoke.handlers.bgc_boundary", "BGCBoundaryHandler"),
}


class _LazyHandler:
    """Deferred handler: imports the concrete handler module on first ``run``.

    Dispatch-side semantics are identical to a real handler — the dispatcher only
    ever calls ``run(request)``.  The concrete module (and its torch import) is
    loaded lazily on the first dispatched request, after validation passed.
    """

    def __init__(self, module, attr):
        self._module = module
        self._attr = attr
        self._handler = None

    def run(self, request):
        if self._handler is None:
            handler_cls = getattr(importlib.import_module(self._module), self._attr)
            self._handler = handler_cls()
        return self._handler.run(request)


# Register the five handlers with the dispatcher (weights + model resolved lazily).
for _task, (_module, _attr) in _HANDLER_FACTORIES.items():
    register(_task, _LazyHandler(_module, _attr))


def run_task(file_path, task):
    req = validate(task, file_path)
    return dispatch(req)


__all__ = [
    "run_task", "parse_prompt", "dispatch", "register", "Report",
    "StructuredRequest", "Task", "task_from_string", "ValidationError", "validate",
]
