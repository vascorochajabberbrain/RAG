"""
Centralized token usage tracker for all OpenAI API calls.

Tracks input/output tokens per model, both per-task and cumulative session totals.
Persists session totals and appends per-step log to .token_usage.log.
"""
import threading
import os
import json
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_PROJECT_ROOT, ".token_usage.log")
_SESSION_FILE = os.path.join(_PROJECT_ROOT, ".token_session.json")


class TokenTracker:
    """Thread-safe singleton that accumulates OpenAI token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        # Session totals: {model: {input: int, output: int}}
        self._session: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        # Current task tokens (reset before each pipeline step)
        self._task: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        # Restore session from disk if available
        self._load_session()

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
            self._save_session()

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

    def log_step(self, step_name: str, collection_name: str = "", solution_id: str = ""):
        """Append per-step token usage to .token_usage.log and persist session."""
        task = self.get_task_usage()
        if not task:
            return
        parts = []
        for model, counts in sorted(task.items()):
            inp, out = counts.get("input", 0), counts.get("output", 0)
            parts.append(f"{model}: {inp:,} in / {out:,} out")
        tokens_str = " | ".join(parts)
        total_in = sum(c.get("input", 0) for c in task.values())
        total_out = sum(c.get("output", 0) for c in task.values())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        coll_label = f" ({collection_name})" if collection_name else ""
        sol_label = f" [{solution_id}]" if solution_id else ""
        line = f"{timestamp} | {step_name}{coll_label}{sol_label} | {tokens_str} | total: {total_in:,} in / {total_out:,} out"

        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[token_tracker] Failed to write log: {e}")

        self._save_session()

    def _save_session(self):
        """Persist session totals to disk."""
        try:
            data = {model: dict(counts) for model, counts in self._session.items()
                    if counts["input"] or counts["output"]}
            with open(_SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_session(self):
        """Restore session totals from disk."""
        try:
            if os.path.exists(_SESSION_FILE):
                with open(_SESSION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for model, counts in data.items():
                    self._session[model]["input"] = counts.get("input", 0)
                    self._session[model]["output"] = counts.get("output", 0)
        except Exception:
            pass


# Module-level singleton
_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    return _tracker


def record_usage(model: str, input_tokens: int, output_tokens: int = 0):
    """Convenience function to record usage."""
    _tracker.record(model, input_tokens, output_tokens)
