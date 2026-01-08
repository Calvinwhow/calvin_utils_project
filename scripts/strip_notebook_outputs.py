#!/usr/bin/env python3
"""Strip outputs and execution counters from Jupyter notebooks in-place."""

from __future__ import annotations

import sys

import nbformat


def strip_outputs(path: str) -> bool:
    """Remove outputs and execution counts; return True if file changed."""
    nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    changed = False

    for cell in nb.cells:
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        metadata = cell.get("metadata", {})
        if metadata.pop("execution", None) is not None:
            changed = True

    if nb.metadata.pop("signature", None) is not None:
        changed = True

    if changed:
        nbformat.write(nb, path)
    return changed


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: strip_notebook_outputs.py <notebook> [<notebook> ...]", file=sys.stderr)
        return 1

    for path in sys.argv[1:]:
        try:
            strip_outputs(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[strip_notebook_outputs] Failed on {path}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
