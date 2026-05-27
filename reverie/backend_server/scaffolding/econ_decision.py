"""Structured economic-action decision for the economic action loop (substrate).

Closes the loop guide §7.9 / economic-action-loop.md asks for: the agent emits a
STRUCTURED list of trades (not free text), which we parse + clean (reusing the
anti-chatter discipline), validate, and execute through the economy. One
decision per agent per game day (cheap relative to ~50 calls/agent/day).

This module is the deterministic half: parse_trade_decisions() (robust to
chat-model output) + execute_decisions() (validate -> economy.trade via the
typed action API). The LLM call + prompt template is the live-wiring half (step
3); the prompt gives FACTS ONLY, no steering (guide §13.4).

Shared substrate: C1 / C2.5 / C3 / C4 use this same affordance (parity §9.1.3);
C2 keeps GA-native actions. strategy_id stays None here; C3 binds
internal_strategy_id into agent_internal (firewall), never raw.
"""
import json
import os
import re
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

try:
  from economy import config as _econ_config
  _CONFIG_RESOURCES = {r.lower() for r in _econ_config.RESOURCES}
except Exception:
  _CONFIG_RESOURCES = set()

from scaffolding import action_api


def _active_resources():
  """Resources the agent may trade RIGHT NOW: the attached EnvironmentModel's
  (de-semanticized alpha_good/... when GA_ENV_CONFIG is set), else the config
  default (semantic food/coffee/...). Sourced at call time so the de-sem swap
  reaches both the decision prompt and the trade validation -- the import-time
  set would freeze the agent to semantic resources and reject alpha_good."""
  try:
    from economy import manager as _mgr
    m = _mgr.get_env_model()
    if m is not None and m.resources:
      return {r.lower() for r in m.resources}
  except Exception:
    pass
  return _CONFIG_RESOURCES

_SIDES = ("buy", "sell")
# Lines that signal "no trades today" -> empty decision (a legal outcome).
_NONE_MARKERS = ("none", "no trade", "no trades", "nothing", "n/a", "no action")
# Chat-model meta / prose to drop (local, focused; mirrors run_gpt_prompt
# _is_chatty_meta discipline without importing that heavy module).
_CHATTY = ("here", "sure", "okay", "ok,", "let me", "i will", "as ",
           "based on", "would you", "feel free", "note:", "**", "```", "?")
# "buy 2 coffee" / "sell 3 food" (qty and item order tolerant). The [^\d-]
# class (not \D) is deliberate: it refuses to skip across a minus sign, so
# "sell -1 coffee" does NOT become "sell 1 coffee" (negatives stay illegal).
_RE_SIDE_QTY_ITEM = re.compile(r"\b(buy|sell)\b[^\d-]*?(\d+)\s+([a-z_]+)")
_RE_SIDE_ITEM_QTY = re.compile(r"\b(buy|sell)\b\s+([a-z_]+)[^\d-]*?(\d+)")


def _norm_one(side, item, qty):
  side = (side or "").lower().strip()
  item = (item or "").lower().strip()
  try:
    qty = int(qty)
  except Exception:
    return None
  if side not in _SIDES or item not in _active_resources() or qty <= 0:
    return None
  return {"item": item, "qty": qty, "side": side}


def _parse_json(text):
  try:
    data = json.loads(text)
  except Exception:
    return None
  if isinstance(data, dict):
    data = data.get("trades", data.get("decisions", []))
  if not isinstance(data, list):
    return None
  out = []
  for d in data:
    if isinstance(d, dict):
      one = _norm_one(d.get("side"), d.get("item"), d.get("qty", d.get("quantity")))
      if one:
        out.append(one)
  return out


def parse_trade_decisions(raw_text):
  """Parse an LLM economic-decision response into a clean list of
  {item, qty, side}. Robust to JSON, line forms, chatty prose, and 'none'.
  Returns [] for an empty/none/garbled decision (a legal no-op). Never raises."""
  if not raw_text or not str(raw_text).strip():
    return []
  text = str(raw_text).strip()

  # 1) Prefer structured JSON if present.
  js = _parse_json(text)
  if js is not None:
    return js

  # 2) Line-by-line, dropping chatty/meta lines; explicit 'none' -> empty.
  out = []
  for line in text.lower().splitlines():
    line = line.strip(" \t-*•.")
    if not line:
      continue
    if any(m in line for m in _NONE_MARKERS) and not re.search(r"\d", line):
      continue  # "no trades today" -> contributes nothing
    if any(c in line for c in _CHATTY):
      continue
    m = _RE_SIDE_QTY_ITEM.search(line) or _RE_SIDE_ITEM_QTY.search(line)
    if not m:
      continue
    side, a, b = m.group(1), m.group(2), m.group(3)
    # group order differs between the two patterns: figure out which is qty
    qty, item = (a, b) if a.isdigit() else (b, a)
    one = _norm_one(side, item, qty)
    if one:
      out.append(one)
  return out


_DECISION_INSTRUCTION = (
    "{name} may buy or sell resources at the market prices above. "
    "Output ONLY the trades, one per line, in EXACTLY this format: "
    "\"buy <qty> <resource>\" or \"sell <qty> <resource>\" "
    "(resource is one of: {resources}). "
    "If no trades, output exactly: none\n"
    "Do NOT explain, greet, or add any other text. Maximum 3 lines.\n"
    "Trades:")


def build_decision_prompt(manager, agent_name, date_str=""):
  """Facts-only economic-decision prompt (guide 13.4: no steering). Reuses the
  perception block (single source) + a neutral affordance + structured-output
  instruction (so parsing is robust)."""
  try:
    perception = manager.perception_str(agent_name, date_str)
  except Exception:
    perception = "[Economic situation]"
  resources = ", ".join(sorted(_active_resources())) or "food, coffee, ingredients"
  return perception + "\n\n" + _DECISION_INSTRUCTION.format(
      name=agent_name, resources=resources)


def run_daily_decision(agent_name, manager, curr_time, *, bind_strategy_id=None):
  """One structured trade decision for an agent (1 LLM call). Builds the prompt,
  calls the chat model, parses + executes, logs typed records to raw/actions.
  Guarded; never raises into the loop. Returns the executed records (or []).

  C3: `bind_strategy_id` records each executed action<->strategy link to
  agent_internal (firewall). The raw/actions strategy_id stays None."""
  try:
    if manager is None:
      return []
    try:
      date_str = curr_time.strftime("%A %B %d")
    except Exception:
      date_str = ""
    prompt = build_decision_prompt(manager, agent_name, date_str)
    try:
      from persona.prompt_template.gpt_structure import ChatGPT_single_request
      raw = ChatGPT_single_request(prompt)
    except Exception:
      return []
    decisions = parse_trade_decisions(raw)
    # raw strategy_id stays None (firewall); binding logged to agent_internal below
    records = execute_decisions(manager, agent_name, decisions, strategy_id=None)
    if bind_strategy_id:
      try:
        from scaffolding import strategy as _strat
        for rec in records:
          _strat.log_binding(agent_name, rec.get("action_id"), bind_strategy_id)
      except Exception:
        pass
    try:
      import telemetry_log
      # Executed trades are observable -> raw/actions.
      for rec in records:
        telemetry_log.log_event("actions", dict(rec, stage="econ_decision"))
      # The decision PROCESS (raw LLM reply, what was proposed/parsed) is
      # internal cognition -> agent_internal/econ_decision (firewall). Logged
      # every time, so 'agent replied none' vs 'proposed-but-rejected' vs
      # 'rambled' is visible without guessing.
      telemetry_log.log_event("econ_decision", {
          "persona": agent_name,
          "raw_response": (str(raw)[:600] if raw is not None else None),
          "parsed": decisions,
          "n_parsed": len(decisions),
          "n_executed": sum(1 for r in records if r.get("ok"))})
    except Exception:
      pass
    return records
  except Exception:
    return []


def execute_decisions(manager, agent_name, decisions, *, location=None,
                      ctx_for=None, strategy_id=None):
  """Validate + execute each trade decision through the typed action API /
  economy. `ctx_for(decision) -> PlanContext` is an optional validator-context
  builder; if omitted, we rely on economy.trade()'s own legality checks (which
  already reject insufficient cash/inventory/unknown resource). Returns a list
  of typed action records (executed and rejected alike, all logged by caller).
  Never raises."""
  records = []
  for d in (decisions or []):
    try:
      rec, ok = action_api.apply_trade(
          manager, agent_name, d["item"], d["qty"], d["side"],
          location=location, strategy_id=strategy_id)
      rec["decision_source"] = "econ_decision"
      records.append(rec)
    except Exception:
      continue
  return records
