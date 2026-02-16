"""
core.py — Re-export shim for backwards compatibility.

The implementation has been split into:
  defaults.py   — DEFAULT_MODEL_A, DEFAULT_MODEL_B
  helpers.py    — _format_reviewer_prompt, _extract_json
  inference.py  — _InferenceMixin (_call, _run_starter, _run_reviewer)
  checkpoint.py — _CheckpointMixin (save_checkpoint, load_checkpoint, reset)
  hyperthink.py — HyperThink (__init__, _log, query)
"""

from .defaults import DEFAULT_MODEL_A, DEFAULT_MODEL_B
from .hyperthink import HyperThink

__all__ = ["DEFAULT_MODEL_A", "DEFAULT_MODEL_B", "HyperThink"]
