"""Post-run diagnostic trace.

One ``DiagTrace`` per worker process. When disabled (the default) every method
is a near-free no-op (a single bool check), so it is safe to leave the calls in
the hot loops permanently. When enabled (via ``--log`` → ``diag.trace``), records
are appended to an in-RAM list and serialised exactly once at ``flush()`` time —
**never** inside a worker loop, so the SCHED_FIFO control loop takes no disk I/O.

The Step-3 diagnostic tool ingests the resulting ``{dir}/{process}.jsonl`` files
offline and derives all latency statistics from the raw timestamps:

    stage_ns = t - in_ts          # this hop's latency
    cum_ns   = t - origin_ns      # glass→here, when origin_ns > 0

Record kinds (``k``):
    "lat"    — per-iteration latency sample: in_ts, origin_ns
    "state"  — periodic worker state snapshot (arbitrary fields)
    "health" — worker health/state transition: state, detail
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any

__all__ = ["DiagTrace", "resolve_trace_dir"]


def resolve_trace_dir(base_logging_dir: str | None) -> str:
    """Create and return a timestamped trace dir under base_logging_dir.

    Falls back to ./quadguide-trace/<ts> when base (e.g. /var/log/quadguide) is
    not writable — common on a bench where the service log dir is root-owned.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base_logging_dir or ".", "trace", ts)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.join(os.getcwd(), "quadguide-trace", ts)
        os.makedirs(d, exist_ok=True)
    return d


class DiagTrace:
    def __init__(
        self,
        process: str,
        *,
        enabled: bool,
        dir: str | None,
        max_rows: int = 0,
    ) -> None:
        self.process = process
        self.enabled = bool(enabled) and dir is not None
        self._dir = dir
        # max_rows == 0 → unbounded; otherwise a ring that bounds RAM on long runs.
        self._rows: deque = deque(maxlen=max_rows) if max_rows > 0 else deque()
        self._flushed = False

    def latency(self, now_ns: int, input_ts_ns: int | None, origin_ns: int) -> None:
        if not self.enabled:
            return
        self._rows.append(("lat", now_ns, input_ts_ns, origin_ns))

    def state(self, now_ns: int, **fields: Any) -> None:
        if not self.enabled:
            return
        self._rows.append(("state", now_ns, fields))

    def health(self, now_ns: int, state: str, detail: str = "") -> None:
        if not self.enabled:
            return
        self._rows.append(("health", now_ns, state, detail))

    def flush(self) -> None:
        """Serialise all buffered records to {dir}/{process}.jsonl, once.

        Safe to call multiple times (e.g. post-loop and again in a finally) — only
        the first call writes. Tolerant of I/O errors so a trace failure never
        takes down a worker shutdown path.
        """
        if not self.enabled or self._flushed:
            return
        self._flushed = True
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = os.path.join(self._dir, f"{self.process}.jsonl")
            with open(path, "w") as f:
                for row in self._rows:
                    f.write(json.dumps(self._encode(row)) + "\n")
        except OSError:
            pass

    def _encode(self, row: tuple) -> dict:
        kind = row[0]
        if kind == "lat":
            _, t, in_ts, origin = row
            return {"t": t, "p": self.process, "k": "lat", "in": in_ts, "org": origin}
        if kind == "state":
            _, t, fields = row
            return {"t": t, "p": self.process, "k": "state", **fields}
        # health
        _, t, state, detail = row
        return {"t": t, "p": self.process, "k": "health", "state": state, "detail": detail}

    def __len__(self) -> int:
        return len(self._rows)
