"""In-context strategy object for C3 (guide 7.6).

C3 = C2.5 + this layer, and ONLY this layer (single-step increment). A strategy
is the agent's in-context cognitive tool, NOT an evaluation channel: every field
here is agent self-reported and lives in agent_internal/strategies.jsonl; the
post-hoc pipeline (2.5) reconstructs lineage/transmission from the raw action log
and never reads these fields.

Firewall (2.6): all self-reported fields are named internal_* / *_self_reported.
The action<->strategy binding is written as internal_strategy_id to
agent_internal/, NOT into raw/actions (whose strategy_id stays null across
C2.5/C3/C4). See C3-strategy-layer.md 3.

This module is pure + deterministic (no LLM/RNG): schema builder, lifecycle state
machine, fitness/mutation bookkeeping, and the compact selection summary injected
into the daily-plan prompt. The LLM only PROPOSES strategy text (wiring step 2);
parsing/cleaning reuses the run_gpt_prompt anti-chatter discipline.
"""
import itertools
import re

_ids = itertools.count(1)

# Per-agent in-run strategy store (agent_internal). Module-level, mirroring the
# economy perception registry pattern, so the loop can propose/select without
# threading a store object everywhere. Each agent -> list[strategy dict].
_STORE = {}

# Lifecycle (guide 7.6): proposal -> trial -> evaluation -> retention / mutation
# / abandonment -> (transmission: C4). `evaluation` is not a stored status; it is
# the act of appending a fitness sample. Transmission is C4, not here.
STATUSES = ("proposed", "trial", "retained", "mutated", "abandoned")
_ALLOWED = {
    "proposed": {"trial", "abandoned"},
    "trial": {"retained", "mutated", "abandoned"},
    "retained": {"trial", "mutated", "abandoned"},
    "mutated": {"abandoned"},      # `mutated` records the parent; the child is new
    "abandoned": set(),            # terminal
}


def new_strategy_id():
  return f"s_{next(_ids)}"


def is_internal_field(name):
  """Firewall check (2.6): agent self-reported strategy fields must be named
  internal_* or *_self_reported so they are never confused with eval fields."""
  return name.startswith("internal_") or name.endswith("_self_reported")


def new_strategy(creator, name, description, *, required_resources=None,
                 required_social_links=None, expected_profit=None,
                 risk_level="medium", evidence_ids=None, created_at=None,
                 parent_id=None):
  """Build a freshly-proposed strategy object (guide 7.6 schema, firewall
  naming). Status starts at 'proposed'."""
  return {
      "internal_strategy_id": new_strategy_id(),
      "name": name,
      "creator": creator,
      "internal_self_reported_parent": parent_id,
      "internal_copied_from_agent": None,   # C4 transmission: teacher agent
      "internal_adoption_time": None,       # C4 transmission: when adopted
      "created_at": created_at,
      "description": description,
      "required_resources": list(required_resources or []),
      "required_social_links": list(required_social_links or []),
      "internal_expected_profit": expected_profit,
      "risk_level": risk_level,
      "evidence_ids": list(evidence_ids or []),
      "status": "proposed",
      "internal_fitness_history_self_reported": [],
      "internal_mutations_self_reported": [],
  }


def can_transition(curr_status, new_status):
  return new_status in _ALLOWED.get(curr_status, set())


def transition(strategy, new_status):
  """Move a strategy to new_status if allowed; returns True on success, False if
  the transition is illegal (caller decides). Never raises."""
  try:
    if can_transition(strategy.get("status"), new_status):
      strategy["status"] = new_status
      return True
  except Exception:
    pass
  return False


def evaluate(strategy, reward):
  """Append a self-reported fitness sample (agent_internal only). No-op on an
  abandoned strategy. Returns the updated history."""
  try:
    if strategy.get("status") != "abandoned":
      strategy["internal_fitness_history_self_reported"].append(reward)
  except Exception:
    pass
  return strategy.get("internal_fitness_history_self_reported", [])


def mutate(parent, name, description, *, note=None, created_at=None, **kwargs):
  """Mark `parent` as mutated and return a NEW child strategy whose
  internal_self_reported_parent points at the parent. The mutation note is
  recorded on BOTH (parent's log + child's history). Child starts 'trial'
  (a mutation is something the agent is trying)."""
  transition(parent, "mutated")
  if note:
    parent["internal_mutations_self_reported"].append(note)
  child = new_strategy(parent.get("creator"), name, description,
                       parent_id=parent.get("internal_strategy_id"),
                       created_at=created_at, **kwargs)
  child["status"] = "trial"
  if note:
    child["internal_mutations_self_reported"].append(note)
  return child


def selection_summary(strategies, max_n=3):
  """Compact, factual summary of the agent's active strategies for injection
  into the daily-plan prompt (guide 11.4 step 3). Only the agent's OWN strategy
  facts -- no experimenter advice/imperatives (guide 13.4, same discipline as
  the C2 economic block). Returns '' if nothing active.

  Only 'trial' and 'retained' strategies are shown (proposed not yet adopted;
  mutated/abandoned are history)."""
  active = [s for s in (strategies or [])
            if s.get("status") in ("trial", "retained")]
  if not active:
    return ""
  lines = ["[Your current strategies]"]
  for s in active[:max_n]:
    res = ", ".join(s.get("required_resources") or []) or "none"
    ep = s.get("internal_expected_profit")
    ep_str = f"; expected profit {ep}" if ep is not None else ""
    lines.append(f"- {s.get('name')} ({s.get('status')}; resources: {res}"
                 f"{ep_str}).")
  return "\n".join(lines)


# --- strategy proposal (LLM) + parsing (guide 7.6 / 11.4 step 2) ------------
# Terse structured output (lesson from econ_decision: chat models ramble unless
# forced). Asking the agent to MAINTAIN a strategy is the C3 mechanism itself,
# not steering -- we never inject a SPECIFIC strategy (guide 13.4).
_PROPOSAL_INSTRUCTION = (
    "{name} keeps a personal economic strategy. Given the situation above, "
    "either propose ONE new strategy to try, or keep the current one(s).\n"
    "Output EXACTLY one line, no other text:\n"
    "STRATEGY: <short name> | <one sentence plan, mention a resource and buy or sell>\n"
    "or, to change nothing, output exactly: keep")

_RE_PROPOSAL = re.compile(r"strategy:\s*(.+?)\s*\|\s*(.+)", re.IGNORECASE)
_PROP_CHATTY = ("here", "sure", "okay", "let me", "i will", "as an",
                "based on", "would you", "```", "note:")


def build_proposal_prompt(agent_name, econ_block, active_summary=""):
  """Facts-only strategy-proposal prompt. econ_block = the economic perception
  string; active_summary = selection_summary() of current strategies (may be '')."""
  parts = [econ_block]
  if active_summary:
    parts.append(active_summary)
  parts.append(_PROPOSAL_INSTRUCTION.format(name=agent_name))
  return "\n\n".join(p for p in parts if p)


def parse_strategy_proposal(raw):
  """Parse an LLM proposal reply into {name, description} or None ('keep'/none/
  garbled -> None). Robust to chatty prose. Never raises."""
  if not raw or not str(raw).strip():
    return None
  for line in str(raw).strip().splitlines():
    line = line.strip(" \t-*•")
    low = line.lower()
    if not line or low in ("keep", "none"):
      continue
    if any(c in low for c in _PROP_CHATTY):
      continue
    m = _RE_PROPOSAL.search(line)
    if m:
      name = m.group(1).strip().strip('"')
      desc = m.group(2).strip().strip('"')
      if name and desc and len(desc) >= 5:
        return {"name": name, "description": desc}
  return None


# --- per-agent store + agent_internal logging ------------------------------

def _log_internal(record):
  try:
    import telemetry_log
    telemetry_log.log_event("strategy", record)   # routes to agent_internal
  except Exception:
    pass


def active_strategies(agent_name):
  return [s for s in _STORE.get(agent_name, [])
          if s.get("status") in ("trial", "retained")]


def active_strategy_id(agent_name):
  """The internal_strategy_id the agent is currently acting under, or None."""
  act = active_strategies(agent_name)
  return act[-1]["internal_strategy_id"] if act else None


def record_strategy(agent_name, strat, event="proposed"):
  """Add/track a strategy for an agent and log the event to agent_internal."""
  _STORE.setdefault(agent_name, []).append(strat)
  _log_internal({"persona": agent_name, "event": event,
                 "strategy": strat})


def log_binding(agent_name, action_id, strategy_id):
  """Record that an action was taken under a strategy -> agent_internal ONLY
  (firewall: never into raw/actions, whose strategy_id stays null)."""
  if not strategy_id:
    return
  _log_internal({"persona": agent_name, "event": "action_binding",
                 "action_id": action_id, "internal_strategy_id": strategy_id})


def update_fitness(agent_name, reward):
  """Append a self-reported fitness sample to the agent's active strategy
  (agent_internal sanity-check only; NOT a metric). Logs the update."""
  act = active_strategies(agent_name)
  if not act:
    return
  s = act[-1]
  evaluate(s, reward)
  _log_internal({"persona": agent_name, "event": "fitness_update",
                 "internal_strategy_id": s["internal_strategy_id"],
                 "reward": reward,
                 "history": s["internal_fitness_history_self_reported"]})


def run_daily_strategy(agent_name, manager, curr_time):
  """C3 proposal+selection (1 LLM call/agent/day). Builds the proposal prompt
  from the economic perception + current strategies, parses, and adds a new
  trial strategy if proposed. Returns the active internal_strategy_id (for
  action binding). Guarded; never raises."""
  try:
    if manager is None:
      return active_strategy_id(agent_name)
    try:
      date_str = curr_time.strftime("%A %B %d")
    except Exception:
      date_str = ""
    try:
      econ_block = manager.perception_str(agent_name, date_str)
    except Exception:
      econ_block = "[Economic situation]"
    summary = selection_summary(_STORE.get(agent_name, []))
    prompt = build_proposal_prompt(agent_name, econ_block, summary)
    try:
      from persona.prompt_template.gpt_structure import ChatGPT_single_request
      raw = ChatGPT_single_request(prompt)
    except Exception:
      raw = None
    proposal = parse_strategy_proposal(raw)
    _log_internal({"persona": agent_name, "event": "proposal_attempt",
                   "raw_response": (str(raw)[:600] if raw is not None else None),
                   "proposed": proposal})
    if proposal:
      strat = new_strategy(agent_name, proposal["name"], proposal["description"],
                           created_at=date_str)
      strat["status"] = "trial"          # selected to try today
      record_strategy(agent_name, strat, event="proposed_trial")
    return active_strategy_id(agent_name)
  except Exception:
    return None
