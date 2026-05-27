"""
Lightweight, side-effect-free telemetry for the GA fork (P2 instrumentation).

Two responsibilities:
  1. Cost meter   -- count LLM/embedding token usage per run (record_llm_call).
  2. Event logger -- append structured JSONL for the cognitive stages (log_event).

HARD RULES (do not break):
  - This module must NEVER raise into the simulation. Every public function
    swallows its own exceptions. It must not change any agent behavior or any
    return value (P2 = instrumentation only).
  - Telemetry firewall (guide 2.6): observable behavior goes to 05-telemetry/raw/
    (evaluator-readable); agent-internal decision process goes to
    05-telemetry/agent_internal/ (evaluator-forbidden). The routing is decided
    here by channel name so the directory discipline holds from P2 onward.

Output layout:
  <research_root>/05-telemetry/raw/<sim_code>/
      events.jsonl       perceived observations (perceive stage)
      actions.jsonl      executed action + movement (execute stage)
      dialogue.jsonl     chats
      world_state.jsonl  positions / world facts (optional)
      llm_calls.jsonl    per-call token usage (ops channel, not an eval metric)
      cost_summary.json  per-run token totals by model
  <research_root>/05-telemetry/agent_internal/<sim_code>/
      retrieve.jsonl     which memories were retrieved (internal)
      reflect.jsonl      generated insights/thoughts (internal)
      plan.jsonl         chosen plan / act_address rationale (internal)
"""
import json
import os
import threading
from datetime import datetime

_LOCK = threading.Lock()

# This file lives at
#   <research_root>/03-reproduction/ga-fork/reverie/backend_server/telemetry_log.py
# so the research root is four directories up.
_RESEARCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_TELEMETRY_ROOT = os.path.join(_RESEARCH_ROOT, "05-telemetry")

# Channels whose content is the agent's internal decision process. These are
# written under agent_internal/ and are off-limits to the evaluator (firewall).
_INTERNAL_CHANNELS = {"retrieve", "reflect", "plan", "econ_decision", "strategy",
                      "transmission"}

_CTX = {"sim_code": "unknown_sim", "step": None, "curr_time": None}
_COST = {}  # model -> {calls, prompt_tokens, completion_tokens, total_tokens}
_ENABLED = os.environ.get("GA_TELEMETRY", "1") not in ("0", "false", "False")


def set_run(sim_code):
  """Set the current simulation code (storage dir name)."""
  _CTX["sim_code"] = sim_code or "unknown_sim"


def set_step(step, curr_time=None):
  """Update the current simulation step + clock (called each main-loop tick)."""
  _CTX["step"] = step
  _CTX["curr_time"] = str(curr_time) if curr_time is not None else None


def _channel_dir(channel):
  base = "agent_internal" if channel in _INTERNAL_CHANNELS else "raw"
  d = os.path.join(_TELEMETRY_ROOT, base, str(_CTX["sim_code"]))
  os.makedirs(d, exist_ok=True)
  return d


def log_event(channel, record):
  """Append one structured record (dict) to <base>/<sim>/<channel>.jsonl.

  Auto-injects ts/sim_code/step/curr_time/channel. Routing (raw vs
  agent_internal) is by channel name. Never raises."""
  if not _ENABLED:
    return
  try:
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "sim_code": _CTX["sim_code"],
        "step": _CTX["step"],
        "curr_time": _CTX["curr_time"],
        "channel": channel,
    }
    if isinstance(record, dict):
      row.update(record)
    else:
      row["value"] = record
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _LOCK:
      path = os.path.join(_channel_dir(channel), f"{channel}.jsonl")
      with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
  except Exception:
    pass


def record_llm_call(call_type, model, usage):
  """Accumulate token usage per model and append one line to llm_calls.jsonl.

  `usage` is the OpenAI/OpenRouter usage object (dict-like) or None. Never
  raises. Some OpenRouter responses omit usage -> counted as zeros."""
  if not _ENABLED:
    return
  try:
    pt = ct = tt = 0
    if usage is not None:
      if hasattr(usage, "get"):
        get = usage.get
      else:
        get = lambda k, d=0: getattr(usage, k, d)
      pt = int(get("prompt_tokens", 0) or 0)
      ct = int(get("completion_tokens", 0) or 0)
      tt = int(get("total_tokens", 0) or (pt + ct))
    with _LOCK:
      agg = _COST.setdefault(model, {"calls": 0, "prompt_tokens": 0,
                                     "completion_tokens": 0, "total_tokens": 0})
      agg["calls"] += 1
      agg["prompt_tokens"] += pt
      agg["completion_tokens"] += ct
      agg["total_tokens"] += tt
    log_event("llm_calls", {"call_type": call_type, "model": model,
                            "prompt_tokens": pt, "completion_tokens": ct,
                            "total_tokens": tt})
  except Exception:
    pass


def cost_summary():
  """Return a copy of accumulated per-model token totals."""
  with _LOCK:
    return {m: dict(v) for m, v in _COST.items()}


def dump_cost_summary():
  """Write the run's token totals to <raw>/<sim>/cost_summary.json. Never raises."""
  if not _ENABLED:
    return
  try:
    summary = {
        "sim_code": _CTX["sim_code"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "by_model": cost_summary(),
        "note": ("token counts from API usage; USD cost requires a per-model "
                 "price table (TODO, fill in 02-protocols or a price config)."),
    }
    with _LOCK:
      path = os.path.join(_channel_dir("llm_calls"), "cost_summary.json")
      with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
  except Exception:
    pass
