"""Typed Action API for C2.5 (guide 7.9).

Turns a natural-language / GA action into a structured, verifiable action record
with `pre_state_hash` / `post_state_hash`, and routes economic actions (`trade`)
to the deterministic economy engine. Benefits (guide 7.9): executable,
verifiable, countable, traceable to strategy, auto-evaluable.

Cross-condition contract (parity 3.1 / 9.1.3):
  * C1 / C2.5 / C3 / C4 share this exact action schema (C1 also uses the typed
    action API, else it is a weakened LLM). C2 keeps GA-native actions
    (heterogeneous flank baseline, adjudicated, disclosed + pre-registered).
  * `strategy_id` in the raw/actions record is **always None across C2.5/C3/C4**
    -- a firewall requirement (2.6: internal_strategy_id is an agent self-reported
    field, must not enter raw eval fields). In C3 the real action<->strategy
    binding is written as internal_strategy_id to agent_internal/strategies, NOT
    here. So C3-vs-C2.5 raw/actions are identical; the diff lives entirely in the
    strategy layer + agent_internal. The null slot is kept only for stable
    cross-condition schema.
  * `trade` is only available in the typed layer (C2.5+); registered as such in
    pre-registration. The economy supports MARKET trade (buy/sell at fixed price,
    no counterparty), so `target_agent` is None for trades until an
    agent-to-agent market exists (second paper). Job income is settled by world
    rules, not an action, so it is parity-equal across all conditions.

OQ-3 (design 8): the state hash covers an ACTION-RELEVANT SUBSET (the acting
agent's cash/inventory/bankrupt + target agent's, if any, + location), NOT the
whole world -- otherwise the hash would churn on unrelated concurrent changes
and break determinism comparisons.

Deterministic: state_hash = sha256 of json.dumps(subset, sort_keys=True). No RNG.
Never raises into the loop: builders return a record; apply_trade reports success.
This module does not write telemetry; callers log the record to raw/actions.
"""
import hashlib
import itertools
import json

ACTION_TYPES = ("move", "interact", "converse", "trade")

_ids = itertools.count(1)


def new_action_id():
  return f"a_{next(_ids)}"


def checkpoint_state():
  """Return the next action-id integer without consuming it."""
  try:
    return {"next_id": int(_ids.__reduce__()[1][0])}
  except Exception:
    return {"next_id": 1}


def restore_checkpoint_state(state):
  """Restore the next action-id integer after an interrupted run."""
  global _ids
  try:
    _ids = itertools.count(max(1, int((state or {}).get("next_id", 1))))
  except Exception:
    _ids = itertools.count(1)


def _agent_state(manager, name):
  """Action-relevant economic slice for one agent, or None if no economy."""
  if manager is None or name is None:
    return None
  try:
    s = manager.snapshot(name)
    return {"cash": s["cash"], "inventory": s["inventory"], "bankrupt": s["bankrupt"]}
  except Exception:
    return None


def state_subset(manager, agent_id, location=None, target_agent=None):
  """The action-relevant subset hashed by pre/post_state_hash (OQ-3)."""
  return {
      "agent": _agent_state(manager, agent_id),
      "target_agent": _agent_state(manager, target_agent),
      "location": location,
  }


def state_hash(subset):
  """Deterministic, cross-run-stable hash of a state subset (12 hex chars)."""
  blob = json.dumps(subset, sort_keys=True, default=str)
  return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def build_action(agent_id, action_type, *, location=None, target_agent=None,
                 item=None, quantity=None, price=None, strategy_id=None,
                 pre_state_hash=None, post_state_hash=None):
  """Assemble a typed action record (guide 7.9 schema). `strategy_id` stays None
  in C2.5; C3 populates it. Field order/keys are the shared cross-condition
  schema -- do not diverge per condition."""
  return {
      "action_id": new_action_id(),
      "agent_id": agent_id,
      "action_type": action_type,
      "target_agent": target_agent,
      "item": item,
      "quantity": quantity,
      "price": price,
      "location": location,
      "strategy_id": strategy_id,          # None in C2.5; bound in C3
      "pre_state_hash": pre_state_hash,
      "post_state_hash": post_state_hash,
  }


def record_ga_action(agent_id, action_type, location, econ_ctx, *,
                     description=None, strategy_id=None):
  """Wrap a GA-native action (move / interact / converse) into the shared typed
  action schema. A GA action does not transact, so the economic state is
  unchanged -> pre_state_hash == post_state_hash (a verifiable no-op on the
  ledger). `econ_ctx` is the agent's current economic slice (or None). Used by
  execute.py so C2.5's raw/actions carries typed records for post-hoc
  reconstruction, symmetrically with C3/C4."""
  subset = {"agent": econ_ctx, "location": location}
  h = state_hash(subset)
  rec = build_action(agent_id, action_type, location=location,
                     strategy_id=strategy_id, pre_state_hash=h, post_state_hash=h)
  if description is not None:
    rec["description"] = description
  return rec


def apply_trade(manager, agent_id, item, quantity, side, *, location=None,
                strategy_id=None):
  """Execute a market trade through the economy engine, capturing the
  action-relevant state before/after. Returns (record, ok).

  `side` is "buy" or "sell". On a rejected trade (insufficient cash/inventory,
  unknown resource, non-positive qty) the economy leaves state unchanged, so
  pre_state_hash == post_state_hash -- a verifiable no-op that is still logged.
  """
  pre = state_subset(manager, agent_id, location=location)
  try:
    ok = bool(manager.trade(agent_id, item, quantity, side)) if manager else False
  except Exception:
    ok = False
  post = state_subset(manager, agent_id, location=location)
  price = None
  try:
    # Log the price the trade actually executed at (manager.price() applies the
    # current-day env-model / phase wave / shock multiplier), not the static
    # base price. The day has not advanced since manager.trade() above, so this
    # is the same unit price that trade() charged. Fall back to base only if the
    # resource is inactive today (price() -> None).
    price = manager.price(item) if manager else None
    if price is None:
      price = manager.cfg.PRICES.get(item) if manager else None
  except Exception:
    price = None
  rec = build_action(
      agent_id, "trade", location=location, item=item,
      quantity=quantity if quantity and quantity > 0 else quantity,
      price=price, strategy_id=strategy_id,
      pre_state_hash=state_hash(pre), post_state_hash=state_hash(post))
  rec["side"] = side
  rec["ok"] = ok
  return rec, ok
