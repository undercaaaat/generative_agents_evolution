"""Phase + shock schedules (P7, guide 6.1-6.6 / 6.5). Deterministic, t=0 frozen.

Two read-only schedules a run harness consults by day index. Neither has any
RNG or runtime researcher input (guide 6.6: 0 interventions during a run):

  PhaseSchedule  — which economic phase (params: price-wave amplitude, whether
                   signals are active, active resources) applies on a given day.
  ShockSchedule  — predefined shocks; exposes a deterministic price multiplier
                   and resource availability (store_closure blocks BUYS).

manager.py optionally consults registered phase + shock schedules in
price()/trade() (guarded; default None -> P3 behavior, existing tests
unchanged). The phase schedule is attached by the run harness before a main
run starts and remains frozen for the whole run.
"""
import json


class PhaseSchedule:
  def __init__(self, phases):
    # Sorted by start_day so phase_for picks the latest phase that has begun.
    self.phases = sorted(phases, key=lambda p: p.get("start_day", 0))

  def phase_for(self, day_index):
    """The phase active on `day_index` (the latest one already started)."""
    if day_index is None:
      return self.phases[0] if self.phases else None
    active = None
    for p in self.phases:
      if p.get("start_day", 0) <= day_index:
        active = p
      else:
        break
    return active or (self.phases[0] if self.phases else None)


class ShockSchedule:
  def __init__(self, shocks):
    self.shocks = list(shocks)

  def active_shocks(self, day_index):
    if day_index is None:
      return []
    out = []
    for s in self.shocks:
      start = s.get("start_day", 0)
      end = start + int(s.get("duration_days", 1))
      if start <= day_index < end:
        out.append(s)
    return out

  def price_multiplier(self, resource, day_index):
    """Deterministic price multiplier on `resource` from active price_spike /
    demand_shift shocks (store_closure does not move price)."""
    mult = 1.0
    for s in self.active_shocks(day_index):
      if s.get("resource") != resource:
        continue
      if s.get("type") in ("price_spike", "demand_shift"):
        mult *= (1.0 + float(s.get("magnitude_pct", 0)) / 100.0)
    return round(mult, 6)

  def is_available(self, resource, day_index):
    """False if a store_closure on `resource` is active (cannot BUY)."""
    for s in self.active_shocks(day_index):
      if s.get("type") == "store_closure" and s.get("resource") == resource:
        return False
    return True

  def wage_multiplier(self, day_index):
    """Deterministic wage multiplier from active wage_cut shocks. Resource-
    independent (cuts the stable income source for all agents), so net worth has
    to be defended by trading. Default (no wage_cut active) -> 1.0."""
    mult = 1.0
    for s in self.active_shocks(day_index):
      if s.get("type") == "wage_cut":
        mult *= (1.0 - float(s.get("magnitude_pct", 0)) / 100.0)
    return round(max(0.0, mult), 6)


def load_phase_schedule(path):
  with open(path, "r", encoding="utf-8") as f:
    return PhaseSchedule(json.load(f).get("phases", []))


def load_shock_schedule(path):
  with open(path, "r", encoding="utf-8") as f:
    return ShockSchedule(json.load(f).get("shocks", []))


def _selftest():
  ok = True

  def check(cond, msg):
    nonlocal ok
    if not cond:
      print("FAIL:", msg); ok = False

  ph = PhaseSchedule([
      {"phase": 1, "name": "basic_survival", "start_day": 0},
      {"phase": 2, "name": "price_fluctuation", "start_day": 10},
      {"phase": 3, "name": "new_resource", "start_day": 20},
  ])
  check(ph.phase_for(0)["phase"] == 1, "day 0 -> phase 1")
  check(ph.phase_for(9)["phase"] == 1, "day 9 -> phase 1")
  check(ph.phase_for(10)["phase"] == 2, "day 10 -> phase 2")
  check(ph.phase_for(25)["phase"] == 3, "day 25 -> phase 3")

  sh = ShockSchedule([
      {"shock_id": "a", "type": "price_spike", "start_day": 14, "duration_days": 2,
       "resource": "ingredients", "magnitude_pct": 80},
      {"shock_id": "b", "type": "store_closure", "start_day": 24, "duration_days": 3,
       "resource": "coffee"},
      {"shock_id": "c", "type": "demand_shift", "start_day": 30, "duration_days": 4,
       "resource": "food", "magnitude_pct": -40},
  ])
  check(sh.price_multiplier("ingredients", 13) == 1.0, "no spike before day 14")
  check(abs(sh.price_multiplier("ingredients", 14) - 1.80) < 1e-9, "ingredients +80% on day 14")
  check(abs(sh.price_multiplier("ingredients", 15) - 1.80) < 1e-9, "spike lasts 2 days")
  check(sh.price_multiplier("ingredients", 16) == 1.0, "spike ends day 16")
  check(sh.is_available("coffee", 24) is False, "coffee unbuyable during closure")
  check(sh.is_available("coffee", 27) is True, "coffee buyable after closure (24+3)")
  check(abs(sh.price_multiplier("food", 31) - 0.60) < 1e-9, "food -40% during demand drop")
  check(sh.price_multiplier("food", 34) == 1.0, "demand shift ends day 34")

  shw = ShockSchedule([
      {"shock_id": "w", "type": "wage_cut", "start_day": 14, "duration_days": 4,
       "magnitude_pct": 80},
  ])
  check(shw.wage_multiplier(13) == 1.0, "no wage cut before day 14")
  check(abs(shw.wage_multiplier(14) - 0.20) < 1e-9, "wage at 20% during cut")
  check(abs(shw.wage_multiplier(17) - 0.20) < 1e-9, "wage cut lasts 4 days")
  check(shw.wage_multiplier(18) == 1.0, "wage restored day 18")
  check(sh.wage_multiplier(14) == 1.0, "price/closure shocks do not touch wage")

  print("schedule selftest: ALL PASS" if ok else "schedule selftest: FAILED")
  return ok


if __name__ == "__main__":
  import sys
  sys.exit(0 if _selftest() else 1)
