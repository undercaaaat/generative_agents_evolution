"""Deterministic, no-API self-test for the P3 economy engine.

Run from backend_server/ with the ga39 interpreter:
    GA_TELEMETRY=0 python economy/selftest.py

Covers: trade primitive (+ rejections), income gating (hour + activity),
multi-day solvency (worker solvent / idler gradual depletion + bankruptcy),
determinism (identical inputs -> identical trajectory), and anti-arbitrage
loophole checks (guide 13.1 / 11.0). Exits non-zero on any failure.
"""
import datetime
import os
import sys

os.environ.setdefault("GA_TELEMETRY", "0")  # don't write telemetry during tests

# Ensure backend_server/ (parent of this economy/ package dir) is importable,
# whether run as `python economy/selftest.py` or `python -m economy.selftest`.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

from economy.manager import EconomyManager


class _Scratch:
  def __init__(self, desc, addr):
    self.act_description = desc
    self.act_address = addr


class _Persona:
  def __init__(self, desc, addr):
    self.scratch = _Scratch(desc, addr)


def _run_days(name, days, work_hours_per_day):
  m = EconomyManager()
  m.ensure_agent(name)
  cash_by_day = []
  for d in range(days):
    for h in range(24):
      cur = datetime.datetime(2023, 2, 13) + datetime.timedelta(days=d, hours=h)
      working = (8 <= h < 8 + work_hours_per_day)
      p = _Persona("serving coffee counter" if working else "sleeping",
                   "Hobbs Cafe" if working else "bed")
      m.step({name: p}, cur, 3600)
    cash_by_day.append(round(m.snapshot(name)["cash"], 1))
  return cash_by_day, m.snapshot(name)


def test_trade_primitive():
  m = EconomyManager()
  assert m.trade("X", "coffee", 2, "buy") is True
  assert m.snapshot("X")["cash"] == 100 - 3 * 2
  assert m.snapshot("X")["inventory"]["coffee"] == 2
  assert m.trade("X", "coffee", 5, "sell") is False    # insufficient inventory
  assert m.trade("X", "coffee", 2, "sell") is True
  assert round(m.snapshot("X")["cash"], 2) == 100.0
  assert m.trade("X", "gold", 1, "buy") is False        # unknown resource


def test_income_gating():
  m = EconomyManager()
  m.ensure_agent("Isabella Rodriguez")
  # working within hours -> paid
  m.accrue_income("Isabella Rodriguez", "serving coffee counter", "Hobbs Cafe", 10, 3600)
  assert round(m.snapshot("Isabella Rodriguez")["cash"], 2) == 104.0
  # working text but outside hours -> not paid
  m.accrue_income("Isabella Rodriguez", "serving coffee counter", "Hobbs Cafe", 3, 3600)
  assert round(m.snapshot("Isabella Rodriguez")["cash"], 2) == 104.0
  # awake hour but sleeping -> not paid
  m.accrue_income("Isabella Rodriguez", "sleeping", "bed", 10, 3600)
  assert round(m.snapshot("Isabella Rodriguez")["cash"], 2) == 104.0


def test_multiday_solvency():
  worker, _ = _run_days("Isabella Rodriguez", 5, 10)
  idler, idler_s = _run_days("Idle Person", 7, 0)
  assert worker[-1] > worker[0] > 100, worker          # worker grows / solvent
  assert idler[0] == 100.0                              # day0: no rollover expense
  assert idler[1] < idler[0]                            # depletes from day1
  assert idler[2] > 0 and idler[3] > 0                  # gradual, not instant
  assert idler_s["bankrupt"] is True                    # eventually bankrupt


def test_determinism():
  def trajectory():
    m = EconomyManager()
    traj = []
    for d in range(3):
      for h in range(24):
        cur = datetime.datetime(2023, 2, 13) + datetime.timedelta(days=d, hours=h)
        w = (8 <= h < 18)
        m.step({"A": _Persona("writing at library" if w else "sleeping",
                              "library" if w else "bed")}, cur, 3600)
        traj.append(round(m.snapshot("A")["cash"], 4))
    return traj
  assert trajectory() == trajectory()


def test_anti_arbitrage():
  m = EconomyManager()
  m.ensure_agent("Z")
  c0 = m.snapshot("Z")["cash"]
  assert m.trade("Z", "coffee", 10, "buy") and m.trade("Z", "coffee", 10, "sell")
  assert round(m.snapshot("Z")["cash"], 4) == round(c0, 4)   # round-trip nets zero
  assert m.trade("Z", "food", 1, "sell") is False            # can't sell unowned
  assert m.trade("Z", "food", 10**9, "buy") is False         # can't overspend
  assert m.trade("Z", "coffee", 0, "buy") is False
  assert m.trade("Z", "coffee", -5, "buy") is False
  assert round(m.snapshot("Z")["cash"], 4) == round(c0, 4)   # no change from rejects


def test_perception():
  from economy.manager import current_perception, _PERCEPTION

  class _Sc:
    def __init__(self, name):
      self.name = name
    def get_str_curr_date_str(self):
      return "Monday February 13"

  class _P:
    def __init__(self, name):
      self.scratch = _Sc(name)

  _PERCEPTION.clear()
  m = EconomyManager()
  assert current_perception("Isabella Rodriguez") is None     # before stamp
  m.stamp_perception({"Isabella Rodriguez": _P("Isabella Rodriguez")}, None)
  blk = current_perception("Isabella Rodriguez")
  assert blk and "Economic situation as of Monday February 13" in blk
  assert "Cash: 100.0" in blk and "Status: solvent" in blk
  assert "Market prices: food 5.0, coffee 3.0, ingredients 2.0." in blk
  # facts only -- no advice/imperative language
  low = blk.lower()
  for banned in ("you should", "consider", "recommend", "try to", "?"):
    assert banned not in low, f"perception block must not steer: {banned!r}"


def test_price_dynamics():
  from economy import config
  # default: flat (amplitude 0) -> price == base (preserves P3 determinism)
  m = EconomyManager()
  assert m.price("coffee", 0) == 3.0 and m.price("food", 7) == 5.0
  saved = config.PRICE_WAVE_AMPLITUDE
  try:
    config.PRICE_WAVE_AMPLITUDE = 0.3
    m2 = EconomyManager()
    prices = [m2.price("coffee", d) for d in range(8)]
    assert max(prices) > 3.0 > min(prices)            # varies around base across days
    assert min(prices) > 0                            # never non-positive
    assert m2.price("coffee", 0) == m2.price("coffee", 4)   # period 4 repeats
    assert m2.price("coffee", 2) == m2.price("coffee", 2)   # deterministic
    # same-day round-trip nets zero (no risk-free arbitrage)
    m2.ensure_agent("T")
    m2._curr_day_index = prices.index(min(prices))
    c0 = m2.snapshot("T")["cash"]
    assert m2.trade("T", "coffee", 2, "buy") and m2.trade("T", "coffee", 2, "sell")
    assert round(m2.snapshot("T")["cash"], 4) == round(c0, 4)
    # buy on a cheap day, sell on an expensive day -> profit (strategy window)
    m2.ensure_agent("U")
    m2._curr_day_index = prices.index(min(prices))
    cu = m2.snapshot("U")["cash"]
    assert m2.trade("U", "coffee", 2, "buy")
    m2._curr_day_index = prices.index(max(prices))
    assert m2.trade("U", "coffee", 2, "sell")
    assert m2.snapshot("U")["cash"] > cu
  finally:
    config.PRICE_WAVE_AMPLITUDE = saved


def main():
  tests = [test_trade_primitive, test_income_gating, test_multiday_solvency,
           test_determinism, test_anti_arbitrage, test_perception,
           test_price_dynamics]
  for t in tests:
    t()
    print(f"  PASS  {t.__name__}")
  print(f"\nALL {len(tests)} ECONOMY SELF-TESTS PASS")


if __name__ == "__main__":
  main()
