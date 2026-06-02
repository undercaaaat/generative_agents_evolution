"""Condition registry: map a single GA_CONDITION selector to feature flags.

Single source of truth so a condition's identity is one name, not a hand-set of
env vars that can drift. Underneath, each scaffolding piece is still an
independent flag (GA_TYPED_MEMORY / GA_RETRIEVAL_RERANK / GA_CONSTRAINT_PLANNER /
GA_TYPED_ACTION / GA_STRATEGY / GA_TRANSMISSION) so each can be unit-toggled.

Design table (C2.5-typed-scaffolding.md 7):
  C1   : economy + perception + typed_action                       (Strong-LLM-Only)
  C2   : economy + perception                                      (GA-Faithful + Econ)
  C2.5 : C2 + typed_memory + retrieval_rerank + constraint_planner + typed_action
  C3   : C2.5 + strategy
  C4   : C3  + transmission

Pre-registered sensitivity axis:
  C2.75 is NOT another preset or a seventh sequential causal condition.
  It is the C2.5/C3 robustness rerun selected with GA_RETRIEVAL_MODE=embed.
  The primary ladder continues to use GA_RETRIEVAL_MODE=econ.

Precedence: an explicitly-set individual flag overrides the condition preset
(lets you probe one piece). GA_CONDITION unset -> all scaffolding off, i.e. the
current validated C2-or-baseline behavior (defaults are safe).

The single source for whether the economy itself is on remains
economy.economy_enabled() (GA_ECONOMY); this module only governs scaffolding.
"""
import os

_FLAGS = ("typed_memory", "retrieval_rerank", "constraint_planner",
          "typed_action", "strategy", "transmission")

# condition name -> set of scaffolding flags that are ON
_PRESETS = {
    "C1":   {"typed_action"},
    "C2":   set(),
    "C2.5": {"typed_memory", "retrieval_rerank", "constraint_planner", "typed_action"},
    "C3":   {"typed_memory", "retrieval_rerank", "constraint_planner", "typed_action",
             "strategy"},
    "C4":   {"typed_memory", "retrieval_rerank", "constraint_planner", "typed_action",
             "strategy", "transmission"},
}

_ENV_VAR = {
    "typed_memory": "GA_TYPED_MEMORY",
    "retrieval_rerank": "GA_RETRIEVAL_RERANK",
    "constraint_planner": "GA_CONSTRAINT_PLANNER",
    "typed_action": "GA_TYPED_ACTION",
    "strategy": "GA_STRATEGY",
    "transmission": "GA_TRANSMISSION",
}


def _truthy(v):
  return v is not None and v not in ("0", "false", "False", "")


def condition_name():
  return os.environ.get("GA_CONDITION", "").strip()


def flag(name):
  """Is scaffolding feature `name` on? Explicit env flag wins; else condition
  preset; else off."""
  if name not in _FLAGS:
    return False
  env = os.environ.get(_ENV_VAR[name])
  if env is not None:                       # explicitly set -> overrides preset
    return _truthy(env)
  preset = _PRESETS.get(condition_name())
  return bool(preset) and name in preset


def active_flags():
  """Dict of all scaffolding flags' current on/off state (for telemetry/debug)."""
  return {name: flag(name) for name in _FLAGS}


# Convenience predicates (readability at call sites).
def typed_memory_on():       return flag("typed_memory")
def retrieval_rerank_on():   return flag("retrieval_rerank")
def constraint_planner_on(): return flag("constraint_planner")
def typed_action_on():       return flag("typed_action")
def strategy_on():           return flag("strategy")
def transmission_on():       return flag("transmission")
