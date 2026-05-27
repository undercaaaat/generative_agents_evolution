"""C5 — Scripted Procedural Baseline (guide 9.1 C5; non-LLM reference point).

C5 is NOT an LLM agent. It is a deterministic, scripted economic policy used as
a reference point (NOT an upper bound): if a scripted ε-greedy + imitation agent
already matches the LLM conditions on the headline metrics, that is honest
evidence the LLM path adds little (guide reverse-hypothesis on C5). Because it
makes no LLM calls, C5 is run by this standalone harness rather than the
LLM-driven reverie loop, but it writes the SAME raw telemetry schema
(raw/actions typed trades + raw/economy + raw/dialogue) so the SAME post-hoc
extractors (strategy_unit / lineage / transmission / transfer / recovery)
process it identically. Self-reported fitness/strategy go to agent_internal
(firewall); raw/actions strategy_id stays None.

Mechanism (guide C5 row):
  - explicit strategy table: one candidate strategy per tradable resource
    ("trade R": buy R when its price is below its running mean, sell when above);
  - ε-greedy: each game day pick the highest-fitness strategy w.p. 1-ε, random
    w.p. ε; fitness = EWMA of net-worth change attributed to that strategy;
  - simple imitation: agents are arranged on a ring; each day an agent "meets"
    its neighbour and, w.p. p_imitate, copies the neighbour's current best
    strategy (logged as observable contact in raw/dialogue).

Fully deterministic given seed (no RNG outside a seeded Random). Supports the
de-semanticized env + shocks via the same EconomyManager registries, so C5 runs
in the transfer arena like the LLM conditions.

CLI:
  python scaffolding/scripted_baseline.py --sim c5-demo --agents 4 --days 12 --seed 7
  # optional: --env <transfer_desem.json> --shock <shock_events.json>
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

from economy import config as econ_config
from economy import manager as econ_mgr
from scaffolding import action_api

try:
  import telemetry_log
except Exception:
  telemetry_log = None

EPSILON = 0.1          # ε-greedy exploration
P_IMITATE = 0.3        # prob of copying neighbour's best strategy on contact
TRADE_QTY = 2
_FITNESS_DECAY = 0.5   # EWMA weight on the new observation
_BASE_DATE = datetime(2023, 2, 13)  # day index 0 (matches GA base sim date)


def _resources(manager):
  m = econ_mgr.get_env_model()
  return list(m.resources) if (m is not None and m.resources) else list(manager.cfg.RESOURCES)


def _networth(manager, name, resources, day):
  s = manager.ensure_agent(name)
  nw = float(s["cash"])
  for r in resources:
    p = manager.price(r, day_index=day)
    nw += float(s["inventory"].get(r, 0)) * (p or 0.0)
  return nw


class _ScriptedAgent:
  def __init__(self, name, resources, rng):
    self.name = name
    self.rng = rng
    # explicit strategy table: one "trade R" strategy per resource.
    self.fitness = {f"trade_{r}": 0.0 for r in resources}
    self.price_mean = {}          # running mean per resource
    self.current = None

  def choose(self, resources):
    if self.rng.random() < EPSILON:
      self.current = self.rng.choice(sorted(self.fitness))
    else:
      # argmax fitness, deterministic tie-break by name.
      self.current = max(sorted(self.fitness), key=lambda k: self.fitness[k])
    return self.current.replace("trade_", "")

  def update_fitness(self, strat, delta):
    self.fitness[strat] = ((1 - _FITNESS_DECAY) * self.fitness[strat]
                           + _FITNESS_DECAY * delta)


def run_scripted_sim(sim_code, n_agents=4, n_days=12, seed=7,
                     env_config=None, shock_config=None, env_model=None):
  import random
  rng = random.Random(seed)

  # Attach env model + shocks (optional) so C5 runs in the same arena.
  if env_model is not None:
    econ_mgr.set_env_model(env_model)
  elif env_config:
    from economy import signals
    econ_mgr.set_env_model(signals.load_env(env_config))
  if shock_config:
    from economy import schedule
    econ_mgr.set_shock_schedule(schedule.load_shock_schedule(shock_config))

  manager = econ_mgr.EconomyManager()
  if telemetry_log is not None:
    telemetry_log.set_run(sim_code)

  resources = _resources(manager)
  names = [f"Agent_{i}" for i in range(n_agents)]
  agents = [_ScriptedAgent(n, resources, random.Random(seed + 100 + i))
            for i, n in enumerate(names)]
  for n in names:
    manager.ensure_agent(n)

  for day in range(n_days):
    curr = _BASE_DATE + timedelta(days=day)
    manager._set_day(curr)
    if telemetry_log is not None:
      telemetry_log.set_step(day, curr)

    for ag in agents:
      nw_before = _networth(manager, ag.name, resources, day)
      r = ag.choose(resources)
      p = manager.price(r, day_index=day) or 0.0
      # running mean of this resource's price (seeded by first obs).
      mean = ag.price_mean.get(r, p)
      ag.price_mean[r] = 0.7 * mean + 0.3 * p
      # policy: buy below mean, sell above mean.
      side = "buy" if p <= mean else "sell"
      rec, ok = action_api.apply_trade(manager, ag.name, r, TRADE_QTY, side)
      rec["decision_source"] = "scripted"
      if telemetry_log is not None:
        telemetry_log.log_event("actions", dict(rec, stage="scripted"))
      nw_after = _networth(manager, ag.name, resources, day)
      ag.update_fitness(ag.current, nw_after - nw_before)
      # self-reported strategy + fitness -> agent_internal (firewall).
      if telemetry_log is not None:
        telemetry_log.log_event("strategy", {
            "persona": ag.name, "internal_strategy_id": ag.current,
            "internal_fitness_self_reported": round(ag.fitness[ag.current], 4)})

    # daily living cost.
    for n in names:
      manager.states[n]["cash"] -= manager.cfg.DAILY_NEED_COST

    # imitation on a ring: agent i meets agent (i+1).
    for i, ag in enumerate(agents):
      nb = agents[(i + 1) % len(agents)]
      if telemetry_log is not None:   # contact is observable -> raw/dialogue
        telemetry_log.log_event("dialogue", {
            "persona": ag.name, "target_agent": nb.name,
            "utterance": "(scripted contact)"})
      if ag.rng.random() < P_IMITATE:
        nb_best = max(sorted(nb.fitness), key=lambda k: nb.fitness[k])
        # copy neighbour's best strategy: boost it in own table.
        ag.fitness[nb_best] = max(ag.fitness[nb_best], nb.fitness[nb_best])
        if telemetry_log is not None:
          telemetry_log.log_event("transmission", {
              "persona": ag.name, "internal_copied_from_agent": nb.name,
              "internal_adopted_strategy": nb_best,
              "internal_adoption_time": str(curr)})

    # economy snapshot per agent (observable -> raw/economy).
    if telemetry_log is not None:
      for n in names:
        telemetry_log.log_event("economy", manager.snapshot(n))

  econ_mgr.set_env_model(None)
  econ_mgr.set_shock_schedule(None)
  return manager


def _selftest():
  ok = True

  def check(c, m):
    nonlocal ok
    if not c:
      print("FAIL:", m); ok = False

  # Determinism: same seed -> same final cash vector; telemetry off for speed.
  prev = os.environ.get("GA_TELEMETRY")
  os.environ["GA_TELEMETRY"] = "0"
  # Non-degenerate economy: a price wave so buy-low/sell-high has real P&L and
  # strategies differ in fitness (flat prices would make every strategy
  # equivalent and the run seed-insensitive).
  from economy import signals

  def _wave_model():
    return signals.EnvironmentModel({
        "env_id": "c5test", "resources": ["food", "coffee", "ingredients"],
        "base_prices": {"food": 5.0, "coffee": 3.0, "ingredients": 2.0},
        "signals": {"s": {"type": "square", "period": 4, "high": 2}},
        "causal_families": [{
            "id": "d", "condition": {"all": [{"signal": "s", "op": ">=", "value": 1}]},
            "effect": {"resource": "coffee", "kind": "demand", "pct": 40}}]})
  try:
    m1 = run_scripted_sim("c5-selftest-a", 4, 10, seed=7, env_model=_wave_model())
    m2 = run_scripted_sim("c5-selftest-b", 4, 10, seed=7, env_model=_wave_model())
    c1 = [round(m1.states[f"Agent_{i}"]["cash"], 6) for i in range(4)]
    c2 = [round(m2.states[f"Agent_{i}"]["cash"], 6) for i in range(4)]
    check(c1 == c2, f"non-deterministic: {c1} vs {c2}")
    m3 = run_scripted_sim("c5-selftest-c", 4, 10, seed=8, env_model=_wave_model())
    c3 = [round(m3.states[f"Agent_{i}"]["cash"], 6) for i in range(4)]
    check(c1 != c3, "different seed should differ (under price dynamics)")
    # agents actually traded (inventory or cash moved from start).
    moved = any(m1.states[f"Agent_{i}"]["cash"] != float(econ_config.INITIAL_CASH)
                for i in range(4))
    check(moved, "no economic activity occurred")
  finally:
    if prev is None:
      os.environ.pop("GA_TELEMETRY", None)
    else:
      os.environ["GA_TELEMETRY"] = prev
  print("scripted_baseline (C5) selftest: ALL PASS" if ok
        else "scripted_baseline (C5) selftest: FAILED")
  return ok


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sim", default="c5-demo")
  ap.add_argument("--agents", type=int, default=4)
  ap.add_argument("--days", type=int, default=12)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--env", default=None)
  ap.add_argument("--shock", default=None)
  ap.add_argument("--selftest", action="store_true")
  a = ap.parse_args()
  if a.selftest:
    sys.exit(0 if _selftest() else 1)
  run_scripted_sim(a.sim, a.agents, a.days, a.seed, a.env, a.shock)
  print(f"scripted C5 run '{a.sim}': {a.agents} agents x {a.days} days "
        f"-> 05-telemetry/raw/{a.sim}/")


if __name__ == "__main__":
  main()
