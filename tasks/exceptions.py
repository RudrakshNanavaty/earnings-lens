"""Task-related errors."""


class TaskNotImplementedError(RuntimeError):
    """Raised when reset() is called for a task that is registered but not implemented yet."""
