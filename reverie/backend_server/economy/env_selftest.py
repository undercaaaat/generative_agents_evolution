"""Aggregate P7 offline self-check (no API, deterministic).

One command that exercises every P7 piece:
  - signals.py engine selftest (signal schedules + F1/F2/F3 causal families);
  - schedule.py phase/shock selftest;
  - config loading for all 07-experiments/configs/transfer-envs/*.json + the
    de-semanticization audit (desem envs must have no natural-language goods);
  - env-model + shock-schedule wiring into EconomyManager (price + inventory
    cap + store-closure), with a default-off regression guard;
  - loophole_check adversarial probe.

Run: GA_TELEMETRY=0 python economy/env_selftest.py
Exit 0 = all pass, 1 = a check failed.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from economy import manager, signals, schedule, loophole_check

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", "..", "..", "..",
                                    "07-experiments", "configs"))


def main():
  ok = True

  def section(name, passed):
    nonlocal ok
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    ok = ok and passed

  # 1-2. Engine + schedule unit selftests.
  section("signals engine", signals._selftest())
  section("phase/shock schedule", schedule._selftest())

  # 3. Config loading + de-semanticization audit.
  cfg_ok = True
  for f in sorted(glob.glob(os.path.join(_CFG, "transfer-envs", "*.json"))):
    try:
      m = signals.load_env(f)
    except Exception as e:
      print(f"    config load FAIL {os.path.basename(f)}: {e}"); cfg_ok = False
      continue
    # desem env <=> no natural-language goods names.
    if m.desemanticized == m.has_natural_language_resource():
      print(f"    audit FAIL {os.path.basename(f)}: desem={m.desemanticized} "
            f"nl={m.has_natural_language_resource()}"); cfg_ok = False
  section("transfer-env configs load + de-semanticization audit", cfg_ok)

  # 3b. False-cue decoy signal (addendum 2026-05-27 2.3): a declared signal that
  # varies saliently but is referenced by NO causal family -> drives nothing.
  decoy_ok = True
  td = signals.load_env(os.path.join(_CFG, "transfer-envs", "transfer_desem.json"))
  referenced = set()
  for fam in td.causal_families:
    for clause in (fam.get("condition", {}).get("all", [])
                   + fam.get("condition", {}).get("any", [])):
      referenced.add(clause.get("signal"))
  for decoy in ["signal_S4"]:
    if decoy not in td.signals:
      print(f"    decoy FAIL: {decoy} not declared"); decoy_ok = False
    if decoy in referenced:
      print(f"    decoy FAIL: {decoy} is referenced by a causal family (not a decoy)")
      decoy_ok = False
    levels = {td.signal_level(decoy, d) for d in range(12)}
    if len(levels) < 2:
      print(f"    decoy FAIL: {decoy} does not vary (levels={levels})"); decoy_ok = False
  section("false-cue decoy signal (varies, drives nothing)", decoy_ok)

  # 4. Wiring into EconomyManager + phase boundaries + default-off guard.
  wire_ok = True
  m = signals.load_env(os.path.join(_CFG, "transfer-envs", "transfer_desem.json"))
  manager.set_env_model(m)
  em = manager.EconomyManager()
  em.ensure_agent("A"); em.states["A"]["cash"] = 1000.0
  # S2 high on day 3 -> beta_good supply -30% -> price 3.0*1.3 = 3.9.
  wire_ok = wire_ok and (em.price("beta_good", day_index=3) == 3.9)
  # F3 inventory cap on alpha_good when S1 AND S2 (day 0: both high) -> cap 3.
  em._curr_day_index = 0
  wire_ok = wire_ok and (em.trade("A", "alpha_good", 3, "buy") is True)
  wire_ok = wire_ok and (em.trade("A", "alpha_good", 1, "buy") is False)
  manager.set_env_model(None)
  manager.set_shock_schedule(None)
  # Frozen main schedule: phase 1 keeps prices flat, phase 2 enables signals,
  # and phase 3 exposes the new tradable tool resource.
  train = signals.load_env(os.path.join(_CFG, "transfer-envs", "train_semantic.json"))
  phases = schedule.load_phase_schedule(os.path.join(_CFG, "phase_scheduler.json"))
  manager.set_env_model(train)
  manager.set_phase_schedule(phases)
  em_phase = manager.EconomyManager()
  em_phase._curr_day_index = 0
  em_phase.ensure_agent("B")
  wire_ok = wire_ok and (em_phase.price("coffee", day_index=0) == 3.0)
  wire_ok = wire_ok and (em_phase.price("tool", day_index=19) is None)
  wire_ok = wire_ok and (em_phase.price("tool", day_index=20) == 4.0)
  em_phase._curr_day_index = 20
  em_phase.ensure_agent("B")
  wire_ok = wire_ok and ("tool" in em_phase.states["B"]["inventory"])
  manager.set_env_model(None)
  manager.set_phase_schedule(None)
  # Default-off regression: plain config behavior restored.
  em2 = manager.EconomyManager()
  wire_ok = wire_ok and (em2._resources() == em2.cfg.RESOURCES)
  wire_ok = wire_ok and (em2.price("coffee") == em2.cfg.PRICES["coffee"])
  section("env + shock + phase wiring into manager (+ default-off guard)", wire_ok)

  # 5. Loophole adversarial probe.
  section("loophole adversarial check", loophole_check.main() == 0)

  print("\nenv_selftest: ALL P7 CHECKS PASS" if ok
        else "\nenv_selftest: FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
