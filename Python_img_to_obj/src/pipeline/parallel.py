"""Job scheduling helpers for folder mode."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable


def resolve_workers(workers) -> int:
    if workers in ("auto", None):
        return max(1, (os.cpu_count() or 2) - 1)
    try:
        return max(1, int(workers))
    except (TypeError, ValueError):
        return 1


def run_jobs(items: Iterable, fn: Callable, workers: int) -> list:
    """Run fn(item) over items. Threads keep it simple and avoid pickling backends.

    Failures are captured per-item: results contain either the return value or an
    exception instance, in input order.
    """
    items = list(items)
    results: list = [None] * len(items)
    if workers <= 1 or len(items) <= 1:
        for i, it in enumerate(items):
            try:
                results[i] = fn(it)
            except Exception as e:  # noqa: BLE001
                results[i] = e
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = e
    return results
