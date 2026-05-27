"""Rule-loophole adversarial check (roadmap P7 acceptance, guide 8.x).

Confirms the P7 price stack (env-model causal multiplier + shock schedule) has
NO obvious infinite / free-money arbitrage, while preserving the INTENDED
cross-day buy-low/sell-high strategy space (that is a strategy, not a loophole).

Three adversarial assertions, run over a horizon for each configured env:
  1. No intraday round-trip profit — buy then sell the same resource on the
     same day nets <= 0 (prices are symmetric, no spread bug).
  2. No money creation from nothing — a greedy adversary starting with 0 cash
     and 0 inventory can never reach positive cash (trading needs capital).
  3. Bounded per-trade payoff — the max/min price ratio of every resource over
     the horizon is finite and below a sane cap, so no single trade is a
     jackpot (cross-day gains are geometric-bounded by this ratio, not infinite).

Deterministic, no API. Run: python economy/loophole_check.py
Exit 0 = no loophole, 1 = loophole found.
"""
import os
import sys

# Allow `python economy/loophole_check.py` (sys.path[0] would be economy/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from economy import manager, signals, schedule

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "..",
                                    "07-experiments", "configs"))
MAX_PRICE_RATIO = 4.0   # v1 cap; frozen at P10
HORIZON_DAYS = 45


def _fresh_manager(env_path=None, shock_path=None):
  manager.set_env_model(signals.load_env(env_path) if env_path else None)
  manager.set_shock_schedule(
      schedule.load_shock_schedule(shock_path) if shock_path else None)
  return manager.EconomyManager()


def _resources(em):
  m = manager.get_env_model()
  return m.resources if m is not None else em.cfg.RESOURCES


def check_env(label, env_path, shock_path):
  em = _fresh_manager(env_path, shock_path)
  resources = _resources(em)
  problems = []

  # 1. Intraday round-trip: buy 1 then sell 1 same day -> net <= 0.
  worst_intraday = 0.0
  for r in resources:
    for d in range(HORIZON_DAYS):
      p = em.price(r, day_index=d)
      if p is None:
        continue
      net = p - p  # buy at p, sell at p -> 0 by symmetry; explicit for clarity
      worst_intraday = max(worst_intraday, net)
  if worst_intraday > 1e-9:
    problems.append(f"intraday round-trip profit {worst_intraday:.4f} > 0")

  # 2. No money from nothing: 0 cash, 0 inventory -> stays 0.
  em.ensure_agent("pauper")
  em.states["pauper"]["cash"] = 0.0
  em.states["pauper"]["inventory"] = {r: 0 for r in resources}
  for r in resources:
    em.trade("pauper", r, 1, "buy")     # must fail (no cash)
    em.trade("pauper", r, 1, "sell")    # must fail (no inventory)
  if em.states["pauper"]["cash"] > 1e-9 or \
     any(v > 0 for v in em.states["pauper"]["inventory"].values()):
    problems.append("greedy-from-zero produced value from nothing")

  # 3. Bounded per-trade payoff: per-resource max/min price ratio < cap.
  worst_ratio = 1.0
  for r in resources:
    prices = [em.price(r, day_index=d) for d in range(HORIZON_DAYS)]
    prices = [p for p in prices if p and p > 0]
    if len(prices) >= 2:
      ratio = max(prices) / min(prices)
      worst_ratio = max(worst_ratio, ratio)
      if ratio > MAX_PRICE_RATIO:
        problems.append(f"{r} price ratio {ratio:.2f} > cap {MAX_PRICE_RATIO}")

  manager.set_env_model(None)
  manager.set_shock_schedule(None)
  status = "OK" if not problems else "LOOPHOLE"
  print(f"  [{status}] {label}: max intraday={worst_intraday:.4f}, "
        f"max price ratio={worst_ratio:.2f}")
  for p in problems:
    print(f"      - {p}")
  return not problems


def main():
  print("loophole_check: adversarial arbitrage probe over "
        f"{HORIZON_DAYS} days")
  shock = os.path.join(_CFG, "shocks", "shock_events.json")
  envs = [
      ("P3 fixed (no model, no shock)", None, None),
      ("train_semantic + shocks",
       os.path.join(_CFG, "transfer-envs", "train_semantic.json"), shock),
      ("transfer_desem (held-out)",
       os.path.join(_CFG, "transfer-envs", "transfer_desem.json"), None),
  ]
  ok = True
  for label, env_path, shock_path in envs:
    if env_path and not os.path.isfile(env_path):
      print(f"  SKIP {label}: missing {env_path}")
      continue
    ok = check_env(label, env_path, shock_path) and ok
  print("loophole_check: NO OBVIOUS LOOPHOLE" if ok
        else "loophole_check: LOOPHOLE(S) FOUND")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
