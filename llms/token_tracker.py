"""
Centralized token usage tracker for all OpenAI API calls.

Tracks input/output tokens per model, both per-task and cumulative session totals.
"""
import threading
from collections import defaultdict


class TokenTracker:
    """Thread-safe singleton that accumulates OpenAI token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        # Session totals: {model: {input: int, output: int}}
        self._session: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        # Current task tokens (reset before each pipeline step)
        self._task: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

    def record(self, model: str, input_tokens: int, output_tokens: int = 0):
        """Record token usage from an API call."""
        with self._lock:
            self._session[model]["input"] += input_tokens
            self._session[model]["output"] += output_tokens
            self._task[model]["input"] += input_tokens
            self._task[model]["output"] += output_tokens

    def reset_task(self):
        """Reset per-task counters (call at start of each pipeline step)."""
        with self._lock:
            self._task = defaultdict(lambda: {"input": 0, "output": 0})

    def reset_session(self):
        """Reset all counters (user clicks reset in UI)."""
        with self._lock:
            self._session = defaultdict(lambda: {"input": 0, "output": 0})
            self._task = defaultdict(lambda: {"input": 0, "output": 0})

    def get_task_usage(self) -> dict:
        """Return per-task token counts."""
        with self._lock:
            return {model: dict(counts) for model, counts in self._task.items() if counts["input"] or counts["output"]}

    def get_session_usage(self) -> dict:
        """Return cumulative session token counts."""
        with self._lock:
            return {model: dict(counts) for model, counts in self._session.items() if counts["input"] or counts["output"]}

    def get_all(self) -> dict:
        """Return both task and session usage."""
        return {
            "task": self.get_task_usage(),
            "session": self.get_session_usage(),
        }


# Module-level singleton
_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    return _tracker


def record_usage(model: str, input_tokens: int, output_tokens: int = 0):
    """Convenience function to record usage."""
    _tracker.record(model, input_tokens, output_tokens)
