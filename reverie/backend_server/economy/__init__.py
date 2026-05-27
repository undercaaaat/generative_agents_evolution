"""Minimal economic layer for the GA fork (P3, Track A "GA + Economy" base).

Design (P3, option A):
  - Deterministic background settlement; agents do NOT perceive money yet.
  - Decoupled from GA agent code; gated by env var GA_ECONOMY (default off) so
    pure-GA / C1 runs are unaffected.
  - Observes each step's activity, settles income/expense, writes economic state
    to telemetry (raw/economy.jsonl). Agent-driven trades arrive with economic
    perception in C2+.
"""
from economy.manager import (  # noqa: F401
    EconomyManager, economy_enabled, current_perception)
