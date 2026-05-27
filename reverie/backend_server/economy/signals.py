"""Deterministic signal time-series + causal-family interpreter (P7, guide 6 / 8.7).

This is the new mechanism P7 adds on top of the P3 economy: an abstract,
config-driven, t=0-frozen causal layer where external SIGNALS drive resource
demand/supply/inventory limits. It is what makes the de-semanticized transfer
test possible — the same engine runs the semantic training environment
(food/coffee/...) and the held-out de-semanticized environment (alpha_good /
signal_S1) so a strategy that "watches a signal -> adjusts inventory -> adjusts
buy/sell" transfers structurally rather than via natural-language priors.

HARD CONSTRAINTS (guide 6.6 / 11.0):
  - Fully DETERMINISTIC: signal levels are a closed-form function of day index
    (square/explicit/const schedules). No RNG, no runtime researcher input.
    Everything is fixed at t=0 by the env config and reproducible per seed.
  - Self-contained: no import of manager.py (manager optionally consults a
    registered model; default none -> P3 behavior, existing tests unchanged).

Causal family (guide 8.7) — a rule mapping a signal condition to a market effect:
  {
    "id": "F1",
    "description": "direction flip: S2 high -> beta_good supply -30%",
    "condition": {"all": [{"signal": "signal_S2", "op": ">=", "value": 1}]},
    "lag_days": 0,                         # effect applies this many days later
    "effect": {"resource": "beta_good", "kind": "supply", "pct": -30}
  }
effect.kind:
  demand  pct>0 -> price up   (price *= 1 + pct/100)
  supply  pct<0 -> price up   (scarcity; price *= 1 - pct/100)
  inventory_cap  effect has "cap": N -> max holdable units while active

The three held-out families are deliberately NON-isomorphic (8.7): F1 flips
direction, F2 swaps the causal end + adds a lag, F3 is a composite (S1 AND S2).
"""
import json

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


class EnvironmentModel:
  """A frozen environment: resources + base prices + signal schedules +
  causal families. Pure function of day index."""

  def __init__(self, d):
    self.env_id = d.get("env_id", "unnamed_env")
    self.desemanticized = bool(d.get("desemanticized", False))
    self.resources = list(d.get("resources", []))
    self.base_prices = dict(d.get("base_prices", {}))
    self.signals = dict(d.get("signals", {}))
    self.causal_families = list(d.get("causal_families", []))

  # -- signals ------------------------------------------------------------
  def signal_level(self, signal, day_index):
    """Deterministic level of `signal` on `day_index` (0 if undefined)."""
    spec = self.signals.get(signal)
    if spec is None or day_index is None or day_index < 0:
      return 0
    kind = spec.get("type", "square")
    if kind == "const":
      return spec.get("level", 0)
    if kind == "explicit":
      levels = spec.get("levels", [])
      if not levels:
        return 0
      if spec.get("repeat", True):
        return levels[day_index % len(levels)]
      return levels[day_index] if day_index < len(levels) else levels[-1]
    # default: square wave — level 1 for the first `high` days of each period.
    period = max(1, int(spec.get("period", 4)))
    high = int(spec.get("high", period // 2))
    phase = int(spec.get("phase", 0))
    return 1 if ((day_index + phase) % period) < high else 0

  # -- causal evaluation --------------------------------------------------
  def _condition_holds(self, condition, day_index):
    if day_index is None or day_index < 0:
      return False
    clauses = condition.get("all") or condition.get("any") or []
    is_any = "any" in condition
    results = []
    for c in clauses:
      lvl = self.signal_level(c["signal"], day_index)
      op = _OPS.get(c.get("op", ">="))
      results.append(bool(op(lvl, c.get("value", 1))) if op else False)
    if not results:
      return False
    return any(results) if is_any else all(results)

  def active_effects(self, day_index):
    """List of (family_id, effect) whose (lagged) condition holds today."""
    out = []
    for fam in self.causal_families:
      lag = int(fam.get("lag_days", 0))
      eff_day = (day_index - lag) if day_index is not None else None
      if eff_day is not None and eff_day >= 0 and \
         self._condition_holds(fam.get("condition", {}), eff_day):
        out.append((fam.get("id", "?"), fam.get("effect", {})))
    return out

  def base_price(self, resource):
    return self.base_prices.get(resource)

  def price_multiplier(self, resource, day_index):
    """Product of all active demand/supply effects on `resource` today."""
    mult = 1.0
    for _fid, eff in self.active_effects(day_index):
      if eff.get("resource") != resource:
        continue
      kind = eff.get("kind")
      pct = float(eff.get("pct", 0))
      if kind == "demand":
        mult *= (1.0 + pct / 100.0)
      elif kind == "supply":
        mult *= (1.0 - pct / 100.0)
    return round(mult, 6)

  def inventory_cap(self, resource, day_index):
    """Tightest active inventory cap for `resource` today, or None."""
    caps = [int(eff.get("cap")) for _f, eff in self.active_effects(day_index)
            if eff.get("kind") == "inventory_cap"
            and eff.get("resource") == resource and eff.get("cap") is not None]
    return min(caps) if caps else None

  def has_natural_language_resource(self):
    """True if any resource/signal name looks like a real-world good (used by
    the de-semanticization audit). De-semanticized envs must return False."""
    banned = ("coffee", "bread", "food", "rain", "festival", "tea", "fruit",
              "drink", "weather", "season", "ingredient", "milk", "snack")
    names = [r.lower() for r in self.resources] + \
            [s.lower() for s in self.signals.keys()]
    return any(any(b in n for b in banned) for n in names)


def load_env(path):
  with open(path, "r", encoding="utf-8") as f:
    return EnvironmentModel(json.load(f))


# ---------------------------------------------------------------------------
# Selftest (deterministic, no API). Run: python economy/signals.py
# ---------------------------------------------------------------------------
def _selftest():
  ok = True

  def check(cond, msg):
    nonlocal ok
    if not cond:
      print("FAIL:", msg)
      ok = False

  # Training env: S_W square (period 4, high 2) -> coffee demand +30%.
  train = EnvironmentModel({
      "env_id": "train", "desemanticized": False,
      "resources": ["food", "coffee", "ingredients"],
      "base_prices": {"food": 5.0, "coffee": 3.0, "ingredients": 2.0},
      "signals": {"signal_W": {"type": "square", "period": 4, "high": 2}},
      "causal_families": [{
          "id": "T", "condition": {"all": [{"signal": "signal_W", "op": ">=", "value": 1}]},
          "effect": {"resource": "coffee", "kind": "demand", "pct": 30}}],
  })
  check(train.signal_level("signal_W", 0) == 1, "W high on day 0")
  check(train.signal_level("signal_W", 2) == 0, "W low on day 2")
  check(abs(train.price_multiplier("coffee", 0) - 1.30) < 1e-9, "coffee +30% when W high")
  check(train.price_multiplier("coffee", 2) == 1.0, "coffee flat when W low")
  check(train.price_multiplier("food", 0) == 1.0, "food unaffected")

  # De-semanticized held-out: three NON-isomorphic families.
  test = EnvironmentModel({
      "env_id": "transfer_desem", "desemanticized": True,
      "resources": ["alpha_good", "beta_good", "gamma_good"],
      "base_prices": {"alpha_good": 5.0, "beta_good": 3.0, "gamma_good": 2.0},
      "signals": {
          "signal_S1": {"type": "square", "period": 6, "high": 3},
          "signal_S2": {"type": "square", "period": 6, "high": 3, "phase": 3},
          "signal_S3": {"type": "explicit", "levels": [0, 0, 1, 0, 0, 0], "repeat": True},
      },
      "causal_families": [
          {"id": "F1", "condition": {"all": [{"signal": "signal_S2", "op": ">=", "value": 1}]},
           "effect": {"resource": "beta_good", "kind": "supply", "pct": -30}},
          {"id": "F2", "lag_days": 2,
           "condition": {"all": [{"signal": "signal_S3", "op": ">=", "value": 1}]},
           "effect": {"resource": "gamma_good", "kind": "demand", "pct": 50}},
          {"id": "F3",
           "condition": {"all": [{"signal": "signal_S1", "op": ">=", "value": 1},
                                 {"signal": "signal_S2", "op": ">=", "value": 1}]},
           "effect": {"resource": "alpha_good", "kind": "inventory_cap", "cap": 3}},
      ],
  })
  # F1 direction flip: S2 high on days 3,4,5 -> beta supply -30% -> price *1.3.
  check(test.signal_level("signal_S2", 3) == 1, "S2 high day 3")
  check(abs(test.price_multiplier("beta_good", 3) - 1.30) < 1e-9, "F1 beta +30% (scarcity)")
  check(test.price_multiplier("beta_good", 0) == 1.0, "F1 inactive day 0")
  # F2 lag: S3 high on day 2 -> effect on gamma at day 4 (2+2), not day 2.
  check(test.price_multiplier("gamma_good", 2) == 1.0, "F2 no same-day effect")
  check(abs(test.price_multiplier("gamma_good", 4) - 1.50) < 1e-9, "F2 lagged +50% at T+2")
  # F3 composite: S1 (days 0-2) AND S2 (days 3-5) never overlap in period 6 ->
  # cap never triggers here; verify the AND logic with a constructed overlap.
  check(test.inventory_cap("alpha_good", 0) is None, "F3 no cap when only S1 high")
  comp = EnvironmentModel({
      "env_id": "x", "resources": ["alpha_good"], "base_prices": {"alpha_good": 5.0},
      "signals": {"signal_S1": {"type": "const", "level": 1},
                  "signal_S2": {"type": "const", "level": 1}},
      "causal_families": [{
          "id": "F3", "condition": {"all": [{"signal": "signal_S1", "op": ">=", "value": 1},
                                            {"signal": "signal_S2", "op": ">=", "value": 1}]},
          "effect": {"resource": "alpha_good", "kind": "inventory_cap", "cap": 3}}],
  })
  check(comp.inventory_cap("alpha_good", 0) == 3, "F3 cap=3 when S1 AND S2 high")

  # De-semanticization audit: abstract names clean, semantic names flagged.
  check(test.has_natural_language_resource() is False, "desem env has no NL goods")
  check(train.has_natural_language_resource() is True, "semantic env flagged as NL")

  print("signals selftest: ALL PASS" if ok else "signals selftest: FAILED")
  return ok


if __name__ == "__main__":
  import sys
  sys.exit(0 if _selftest() else 1)
