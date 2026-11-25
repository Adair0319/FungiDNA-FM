"""Task dispatcher — routes a validated StructuredRequest to its handler."""
from .request import Task

_HANDLERS = {}


def register(task, handler):
    _HANDLERS[task] = handler


def dispatch(request):
    handler = _HANDLERS.get(request.task)
    if handler is None:
        raise RuntimeError(f"no handler registered for task: {request.task.value}")
    return handler.run(request)
