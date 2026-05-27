"""Deterministic, no-API self-test for the C2.5 constraint planner / validator.

Run from backend_server/ with the ga39 interpreter:
    GA_TELEMETRY=0 python scaffolding/selftest.py

Covers guide 7.8 acceptance (roadmap P4 acceptance point 2): a well-formed plan
passes, and each illegal-plan category is correctly rejected. Exits non-zero on
any failure. No LLM / no RNG -> reproducible.
"""
import os
import sys

os.environ.setdefault("GA_TELEMETRY", "0")

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

from scaffolding.validator import PlanContext, validate_plan
from scaffolding import action_api
from scaffolding import typed_memory
from scaffolding import retrieval
from scaffolding import conditions
from scaffolding import strategy
from scaffolding import econ_decision
from scaffolding import transmission
from economy.manager import EconomyManager


class _FakeNode:
  """Minimal stand-in for a GA ConceptNode (offline test only)."""
  def __init__(self, node_id, description, ntype="event", poignancy=5, keywords=None):
    self.node_id = node_id
    self.description = description
    self.type = ntype
    self.poignancy = poignancy
    self.keywords = keywords or []
    self.filling = []


def _legal_plan():
  return {
      "plan_id": "p_ok",
      "agent_id": "maria",
      "goal": "sell breakfast packs",
      "start_min": 420,   # 07:00
      "end_min": 540,     # 09:00
      "location": "Town Market",
      "required_resources": ["food", "stall"],
      "preconditions": ["stall_available"],
  }


def _legal_ctx():
  return PlanContext(
      busy_intervals=[(0, 360)],              # asleep 00:00-06:00, no clash
      known_locations={"Town Market", "Hobbs Cafe"},
      open_locations={"Town Market", "Hobbs Cafe"},
      reachable_locations={"Town Market", "Hobbs Cafe"},
      inventory={"food": 3},
      facilities={"stall", "counter"},
      available_facilities={"stall", "counter"},
      satisfied_preconditions={"stall_available", "has_inventory"})


def _assert_fail(plan, ctx, expect_check):
  res = validate_plan(plan, ctx)
  assert not res.ok, f"expected fail but passed: {plan.get('goal')}"
  checks = {c for c, _ in res.failures}
  assert expect_check in checks, \
      f"expected check {expect_check!r} in failures, got {checks} ({res.reason()})"


def test_legal_plan_passes():
  res = validate_plan(_legal_plan(), _legal_ctx())
  assert res.ok, f"legal plan should pass, got: {res.reason()}"
  assert res.result == "pass"
  assert res.as_dict()["validator_result"] == "pass"


def test_time_conflict():
  p = _legal_plan()
  ctx = _legal_ctx()
  ctx.busy_intervals = [(480, 600)]            # 08:00-10:00 clashes with 07:00-09:00
  _assert_fail(p, ctx, "time")


def test_time_inverted():
  p = _legal_plan()
  p["start_min"], p["end_min"] = 540, 420      # end before start
  _assert_fail(p, _legal_ctx(), "time")


def test_location_closed():
  ctx = _legal_ctx()
  ctx.open_locations = {"Hobbs Cafe"}          # Town Market not open
  _assert_fail(_legal_plan(), ctx, "location_open")


def test_resource_missing():
  ctx = _legal_ctx()
  ctx.inventory = {"food": 0}                   # no food on hand
  _assert_fail(_legal_plan(), ctx, "resources")


def test_unknown_location():
  ctx = _legal_ctx()
  ctx.known_locations = {"Hobbs Cafe"}          # agent doesn't know Town Market
  _assert_fail(_legal_plan(), ctx, "knows_location")


def test_no_path():
  ctx = _legal_ctx()
  ctx.reachable_locations = {"Hobbs Cafe"}      # no path to Town Market
  _assert_fail(_legal_plan(), ctx, "path")


def test_world_rule_unknown_object():
  p = _legal_plan()
  p["required_resources"] = ["food", "unobtanium"]   # not a good, not a facility
  _assert_fail(p, _legal_ctx(), "world_rules")


def test_unmet_precondition():
  p = _legal_plan()
  p["preconditions"] = ["stall_available", "has_license"]   # license not satisfied
  _assert_fail(p, _legal_ctx(), "world_rules")


def test_facility_unavailable():
  ctx = _legal_ctx()
  ctx.available_facilities = {"counter"}        # stall exists but not available now
  _assert_fail(_legal_plan(), ctx, "resources")


def test_collects_multiple_failures():
  # An empty context for a demanding plan should fail several checks at once,
  # proving the validator reports all problems, not just the first.
  ctx = PlanContext(known_locations=set(), open_locations=set(),
                    reachable_locations=set(), inventory={})
  res = validate_plan(_legal_plan(), ctx)
  assert not res.ok
  assert len(res.failures) >= 3, f"expected multiple failures, got {res.reason()}"


def test_unconstrained_dimensions_pass():
  # None set fields mean "not modeled" -> those checks pass (validator stays
  # honest: only fail on dimensions we have ground truth for).
  p = {"goal": "wander", "location": "Mystery Spot", "required_resources": []}
  res = validate_plan(p, PlanContext())   # all defaults / None
  assert res.ok, f"unconstrained plan should pass, got: {res.reason()}"


# --- typed action API (guide 7.9) ------------------------------------------

def test_action_schema_fields():
  rec = action_api.build_action("isabella", "move", location="Hobbs Cafe")
  for k in ("action_id", "agent_id", "action_type", "target_agent", "item",
            "quantity", "price", "location", "strategy_id",
            "pre_state_hash", "post_state_hash"):
    assert k in rec, f"missing schema field {k!r}"
  assert rec["strategy_id"] is None      # C2.5: slot always empty


def test_action_ids_unique():
  ids = {action_api.build_action("a", "move")["action_id"] for _ in range(50)}
  assert len(ids) == 50


def test_trade_success_changes_post_hash():
  m = EconomyManager()
  m.ensure_agent("isabella")
  rec, ok = action_api.apply_trade(m, "isabella", "coffee", 2, "buy",
                                   location="Hobbs Cafe")
  assert ok is True
  assert rec["action_type"] == "trade" and rec["side"] == "buy"
  assert rec["price"] == 3.0 and rec["target_agent"] is None
  assert rec["strategy_id"] is None       # C2.5
  assert rec["pre_state_hash"] != rec["post_state_hash"]   # state really changed


def test_trade_reject_is_verifiable_noop():
  m = EconomyManager()
  m.ensure_agent("z")
  rec, ok = action_api.apply_trade(m, "z", "coffee", 10**9, "buy")  # can't afford
  assert ok is False
  assert rec["pre_state_hash"] == rec["post_state_hash"]   # no-op, but logged
  rec2, ok2 = action_api.apply_trade(m, "z", "gold", 1, "buy")      # unknown good
  assert ok2 is False and rec2["pre_state_hash"] == rec2["post_state_hash"]


def test_state_hash_deterministic():
  m1, m2 = EconomyManager(), EconomyManager()
  m1.ensure_agent("a"); m2.ensure_agent("a")
  h1 = action_api.state_hash(action_api.state_subset(m1, "a", location="X"))
  h2 = action_api.state_hash(action_api.state_subset(m2, "a", location="X"))
  assert h1 == h2, "identical state must hash identically (reproducibility)"


def test_no_economy_does_not_crash():
  rec, ok = action_api.apply_trade(None, "a", "coffee", 1, "buy")
  assert ok is False and rec["action_type"] == "trade"   # graceful with no economy


def test_record_ga_action_is_ledger_noop():
  ctx = {"cash": 100, "daily_cost": 20, "bankrupt": False}
  rec = action_api.record_ga_action("maria", "move", "the Ville:Dorm:room:bed",
                                    ctx, description="sleeping @ bed")
  assert rec["action_type"] == "move" and rec["strategy_id"] is None
  assert rec["pre_state_hash"] == rec["post_state_hash"]   # GA action: no transaction
  assert rec["description"] == "sleeping @ bed"
  # graceful with no economic context (economy off)
  rec2 = action_api.record_ga_action("maria", "converse", "loc", None)
  assert rec2["pre_state_hash"] == rec2["post_state_hash"]


# --- typed memory (guide 7.3) ----------------------------------------------

def test_classify_economic():
  n = _FakeNode("node_1", "Maria sold coffee and earned good income at the cafe")
  assert typed_memory.classify(n) == "economic"
  n2 = _FakeNode("node_2", "Klaus thought about the price of ingredients", "thought")
  assert typed_memory.classify(n2) == "economic"   # economic content wins over thought


def test_classify_social_semantic_episodic():
  assert typed_memory.classify(_FakeNode("c", "talking to Isabella", "chat")) == "social"
  assert typed_memory.classify(_FakeNode("t", "people are generally kind", "thought")) == "semantic"
  assert typed_memory.classify(_FakeNode("e", "Maria is walking to the park")) == "episodic"
  assert typed_memory.classify(_FakeNode("s", "Maria met a friend in the park")) == "social"


def test_extract_resources():
  n = _FakeNode("n", "bought coffee and food at the market")
  res = typed_memory.extract_resources(n)
  assert "coffee" in res and "food" in res and "ingredients" not in res


def test_annotate_schema_and_confidence():
  n = _FakeNode("node_9", "sold coffee", poignancy=8)
  ann = typed_memory.annotate(n)
  for k in ("memory_id", "mem_type", "resources", "money_delta", "confidence",
            "source", "evidence_ids", "valid_until"):
    assert k in ann
  assert ann["memory_id"] == "node_9" and ann["mem_type"] == "economic"
  assert ann["confidence"] == 0.8 and ann["money_delta"] == 0
  # poignancy clamped to [0.1, 1.0]
  assert typed_memory.annotate(_FakeNode("x", "x", poignancy=0))["confidence"] == 0.1
  assert typed_memory.annotate(_FakeNode("y", "y", poignancy=99))["confidence"] == 1.0


# --- evidence-aware retrieval rerank (guide 7.4) ---------------------------

def test_rerank_identity_when_economy_off():
  m = {"node_1": 1.0, "node_2": 2.0}
  idn = {"node_1": _FakeNode("node_1", "sold coffee"), "node_2": _FakeNode("node_2", "walked")}
  assert retrieval.economic_relevance_rerank(m, idn, None) == m   # identity


def test_rerank_boosts_economic_and_urgency():
  idn = {"e": _FakeNode("e", "sold coffee for profit"),
         "p": _FakeNode("p", "took a walk in the park")}
  base = {"e": 1.0, "p": 1.0}
  calm = retrieval.economic_relevance_rerank(base, idn, {"cash": 100, "daily_cost": 20})
  urgent = retrieval.economic_relevance_rerank(base, idn, {"cash": 5, "daily_cost": 20})
  assert calm["e"] > base["e"] and calm["p"] == 1.0        # economic node boosted
  assert urgent["e"] > calm["e"]                           # urgency boosts more
  assert urgent["p"] == 1.0                                # non-economic untouched


def test_rerank_goal_resource_overlap():
  idn = {"c": _FakeNode("c", "coffee demand was high"),
         "f": _FakeNode("f", "food demand was high")}
  base = {"c": 1.0, "f": 1.0}
  out = retrieval.economic_relevance_rerank(
      base, idn, {"cash": 100, "daily_cost": 20, "goal_resources": ["coffee"]})
  assert out["c"] > out["f"]   # node sharing the goal resource ranks higher


def test_contradiction_filter_drops_lower_confidence():
  # Two economic memories about coffee with opposite polarity; the higher-
  # confidence one is the belief, the lower-confidence opposite is dropped.
  belief = _FakeNode("belief", "coffee sold well, big profit", poignancy=9)
  contra = _FakeNode("contra", "coffee was a loss, slow sales", poignancy=3)
  idn = {"belief": belief, "contra": contra}
  filtered, dropped = retrieval.contradiction_filter({"belief": 1.0, "contra": 1.0}, idn)
  assert dropped == ["contra"] and "contra" not in filtered and "belief" in filtered


def test_contradiction_filter_noop_single_polarity():
  a = _FakeNode("a", "coffee sold well, profit", poignancy=9)
  b = _FakeNode("b", "coffee demand grew, more income", poignancy=3)
  filtered, dropped = retrieval.contradiction_filter({"a": 1.0, "b": 1.0}, {"a": a, "b": b})
  assert dropped == [] and len(filtered) == 2   # same polarity -> keep both


def test_apply_rerank_combined():
  idn = {"belief": _FakeNode("belief", "coffee profit, sold well", poignancy=9),
         "contra": _FakeNode("contra", "coffee loss, slow", poignancy=3),
         "walk": _FakeNode("walk", "walked in the park")}
  out, dropped = retrieval.apply_rerank({"belief": 1.0, "contra": 1.0, "walk": 1.0},
                                        idn, {"cash": 5, "daily_cost": 20})
  assert "contra" in dropped and "contra" not in out
  assert out["belief"] > out["walk"]   # economic + urgent boost beats neutral


# --- condition registry (design 7) -----------------------------------------

def _with_env(env, fn):
  saved = {k: os.environ.get(k) for k in env}
  try:
    for k, v in env.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v
    return fn()
  finally:
    for k, v in saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def test_condition_presets():
  # C2.5 turns on the four scaffolding pieces, not strategy/transmission.
  def check_c25():
    f = conditions.active_flags()
    assert f["typed_memory"] and f["retrieval_rerank"]
    assert f["constraint_planner"] and f["typed_action"]
    assert not f["strategy"] and not f["transmission"]
  _with_env({"GA_CONDITION": "C2.5", "GA_TYPED_MEMORY": None,
             "GA_RETRIEVAL_RERANK": None, "GA_CONSTRAINT_PLANNER": None,
             "GA_TYPED_ACTION": None, "GA_STRATEGY": None,
             "GA_TRANSMISSION": None}, check_c25)


def test_condition_c3_adds_only_strategy():
  # The C3-vs-C2.5 diff is exactly the strategy flag (single-step increment).
  def diff():
    c25 = _with_env({"GA_CONDITION": "C2.5"}, conditions.active_flags)
    c3 = _with_env({"GA_CONDITION": "C3"}, conditions.active_flags)
    changed = [k for k in c25 if c25[k] != c3[k]]
    assert changed == ["strategy"], changed
  _with_env({k: None for k in conditions._ENV_VAR.values()}, diff)


def test_c2_and_c1_scaffolding():
  def check():
    c2 = _with_env({"GA_CONDITION": "C2"}, conditions.active_flags)
    assert not any(c2.values())                       # C2: no scaffolding
    c1 = _with_env({"GA_CONDITION": "C1"}, conditions.active_flags)
    assert c1["typed_action"] and not c1["typed_memory"]   # C1 shares typed action only
  _with_env({k: None for k in conditions._ENV_VAR.values()}, check)


def test_explicit_flag_overrides_preset():
  def check():
    on = _with_env({"GA_CONDITION": "C2", "GA_TYPED_ACTION": "1"},
                   conditions.typed_action_on)
    assert on is True                                 # explicit flag beats C2 preset
  _with_env({k: None for k in conditions._ENV_VAR.values()}, check)


def test_no_condition_all_off():
  def check():
    assert not any(conditions.active_flags().values())
  _with_env({k: None for k in list(conditions._ENV_VAR.values()) + ["GA_CONDITION"]},
            check)


# --- C3 strategy layer (guide 7.6) -----------------------------------------

def test_strategy_schema_firewall_naming():
  s = strategy.new_strategy("isabella", "rainy coffee", "sell coffee on rainy eves",
                            required_resources=["coffee"], expected_profit=35)
  assert s["status"] == "proposed" and s["creator"] == "isabella"
  # every self-reported field must be firewall-named (internal_/_self_reported)
  selfreport = ("internal_strategy_id", "internal_self_reported_parent",
                "internal_expected_profit", "internal_fitness_history_self_reported",
                "internal_mutations_self_reported")
  for k in selfreport:
    assert k in s and strategy.is_internal_field(k), k
  # raw-safe descriptive fields are not flagged internal
  assert not strategy.is_internal_field("name")


def test_strategy_ids_unique():
  ids = {strategy.new_strategy("a", "n", "d")["internal_strategy_id"] for _ in range(40)}
  assert len(ids) == 40


def test_strategy_lifecycle_transitions():
  s = strategy.new_strategy("a", "n", "d")
  assert strategy.transition(s, "trial") and s["status"] == "trial"
  assert not strategy.transition(s, "proposed")        # illegal backward
  assert strategy.transition(s, "retained")
  assert strategy.transition(s, "abandoned")
  assert not strategy.transition(s, "trial")           # terminal


def test_strategy_evaluate_appends_fitness():
  s = strategy.new_strategy("a", "n", "d")
  strategy.evaluate(s, 12); strategy.evaluate(s, -5)
  assert s["internal_fitness_history_self_reported"] == [12, -5]
  strategy.transition(s, "abandoned")
  strategy.evaluate(s, 99)                              # no-op once abandoned
  assert s["internal_fitness_history_self_reported"] == [12, -5]


def test_strategy_mutate_links_parent():
  p = strategy.new_strategy("a", "v1", "d", expected_profit=10)
  strategy.transition(p, "trial")
  child = strategy.mutate(p, "v2", "d2", note="added weather check")
  assert p["status"] == "mutated"
  assert "added weather check" in p["internal_mutations_self_reported"]
  assert child["status"] == "trial"
  assert child["internal_self_reported_parent"] == p["internal_strategy_id"]
  assert child["internal_strategy_id"] != p["internal_strategy_id"]


def test_strategy_selection_summary_factual():
  s1 = strategy.new_strategy("a", "rainy coffee", "d", required_resources=["coffee"],
                             expected_profit=35)
  strategy.transition(s1, "trial")
  s2 = strategy.new_strategy("a", "proposed one", "d")   # proposed -> excluded
  out = strategy.selection_summary([s1, s2])
  assert "rainy coffee" in out and "coffee" in out
  assert "proposed one" not in out                       # only trial/retained shown
  # factual only -- no steering language
  low = out.lower()
  for banned in ("you should", "consider", "recommend", "try to", "?"):
    assert banned not in low
  assert strategy.selection_summary([]) == ""            # nothing active -> empty


# --- economic action loop: structured trade decision (substrate) -----------

def test_parse_trade_json():
  txt = '[{"item":"coffee","qty":2,"side":"buy"},{"item":"food","quantity":1,"side":"sell"}]'
  out = econ_decision.parse_trade_decisions(txt)
  assert out == [{"item": "coffee", "qty": 2, "side": "buy"},
                 {"item": "food", "qty": 1, "side": "sell"}]


def test_parse_trade_lines():
  out = econ_decision.parse_trade_decisions("buy 2 coffee\nsell 3 food")
  assert {"item": "coffee", "qty": 2, "side": "buy"} in out
  assert {"item": "food", "qty": 3, "side": "sell"} in out
  # item-then-qty order also works
  assert econ_decision.parse_trade_decisions("sell food 4") == \
      [{"item": "food", "qty": 4, "side": "sell"}]


def test_parse_trade_none_and_chatty():
  assert econ_decision.parse_trade_decisions("none") == []
  assert econ_decision.parse_trade_decisions("No trades today.") == []
  assert econ_decision.parse_trade_decisions("") == []
  # chatty prose with no structured trade -> empty (anti-pollution)
  assert econ_decision.parse_trade_decisions(
      "Sure! Here is what Maria would do based on her cash...") == []


def test_parse_trade_rejects_illegal():
  # unknown resource, zero/neg qty, bad side -> dropped
  out = econ_decision.parse_trade_decisions(
      "buy 2 gold\nbuy 0 coffee\ntrade 5 food\nsell -1 coffee")
  assert out == []


def test_execute_decisions_changes_state():
  m = EconomyManager()
  m.ensure_agent("maria")
  recs = econ_decision.execute_decisions(
      m, "maria", [{"item": "coffee", "qty": 2, "side": "buy"}],
      location="Hobbs Cafe")
  assert len(recs) == 1 and recs[0]["ok"] is True
  assert recs[0]["decision_source"] == "econ_decision"
  assert m.snapshot("maria")["inventory"]["coffee"] == 2
  assert m.snapshot("maria")["cash"] == 100 - 3 * 2
  # illegal decision executes as a verifiable no-op record (pre==post)
  recs2 = econ_decision.execute_decisions(
      m, "maria", [{"item": "coffee", "qty": 10**9, "side": "buy"}])
  assert recs2[0]["ok"] is False
  assert recs2[0]["pre_state_hash"] == recs2[0]["post_state_hash"]


# --- C3 strategy proposal + store + binding (guide 7.6 / 11.4) -------------

def test_parse_strategy_proposal():
  ok = strategy.parse_strategy_proposal(
      "STRATEGY: rainy coffee | buy coffee in the morning and sell it later")
  assert ok == {"name": "rainy coffee",
                "description": "buy coffee in the morning and sell it later"}
  assert strategy.parse_strategy_proposal("keep") is None
  assert strategy.parse_strategy_proposal("none") is None
  assert strategy.parse_strategy_proposal("") is None
  # chatty prose with no STRATEGY line -> None
  assert strategy.parse_strategy_proposal(
      "Sure! Here is what I think Maria should do...") is None


def test_strategy_store_and_active():
  strategy._STORE.clear()
  s = strategy.new_strategy("maria", "n", "buy coffee and sell food")
  s["status"] = "trial"
  strategy.record_strategy("maria", s, event="proposed_trial")
  assert strategy.active_strategy_id("maria") == s["internal_strategy_id"]
  assert len(strategy.active_strategies("maria")) == 1
  # selection summary reflects the stored trial strategy
  summ = strategy.selection_summary(strategy._STORE.get("maria", []))
  assert "n" in summ and "[Your current strategies]" in summ
  strategy._STORE.clear()


def test_strategy_proposal_prompt_factual():
  prompt = strategy.build_proposal_prompt(
      "Maria", "[Economic situation]\nCash: 100.0.", active_summary="")
  low = prompt.lower()
  assert "strategy:" in low                          # asks for structured line
  for banned in ("you should", "consider selling", "recommend"):
    assert banned not in low                          # no specific-strategy steering


def test_strategy_update_fitness_internal():
  strategy._STORE.clear()
  s = strategy.new_strategy("klaus", "n", "sell ingredients daily")
  s["status"] = "trial"
  strategy.record_strategy("klaus", s)
  strategy.update_fitness("klaus", 12)
  strategy.update_fitness("klaus", -3)
  assert s["internal_fitness_history_self_reported"] == [12, -3]
  strategy._STORE.clear()


# --- C4 transmission (guide 11.5) ------------------------------------------

def test_parse_transmission_decision():
  assert transmission.parse_transmission_decision("adopt") == ("adopt", None)
  assert transmission.parse_transmission_decision("ignore") == ("ignore", None)
  assert transmission.parse_transmission_decision(
      "adapt: buy coffee only on cold days") == ("adapt", "buy coffee only on cold days")
  assert transmission.parse_transmission_decision("") is None
  # chatty prose with no decision -> None
  assert transmission.parse_transmission_decision(
      "Well, Maria might think about it...") is None


def test_transmission_adopt_records_lineage():
  strategy._STORE.clear()
  teacher = strategy.new_strategy("isabella", "rainy coffee",
                                  "sell coffee on cold days", required_resources=["coffee"])
  child = transmission.adopt("maria", "isabella", teacher, "Tuesday February 14")
  assert child["internal_copied_from_agent"] == "isabella"          # who taught it
  assert child["internal_adoption_time"] == "Tuesday February 14"
  assert child["internal_self_reported_parent"] == teacher["internal_strategy_id"]
  assert child["status"] == "trial"
  assert child in strategy._STORE["maria"]                          # in learner's store
  assert child["internal_strategy_id"] != teacher["internal_strategy_id"]
  strategy._STORE.clear()


def test_transmission_adapt_is_mutation():
  strategy._STORE.clear()
  teacher = strategy.new_strategy("isabella", "rainy coffee", "sell coffee on cold days")
  child = transmission.adopt("klaus", "isabella", teacher, "day 5",
                             adapt_desc="sell coffee AND food on cold days")
  assert child["description"] == "sell coffee AND food on cold days"
  assert any("adapted:" in m for m in child["internal_mutations_self_reported"])
  assert child["internal_copied_from_agent"] == "isabella"          # adaptation still tracks source
  strategy._STORE.clear()


def test_transmission_run_ignore_and_adopt_offline():
  strategy._STORE.clear()
  teacher = strategy.new_strategy("isabella", "rainy coffee", "sell coffee on cold days")
  # offline llm stub: learner ignores
  out = transmission.run_transmission("maria", "isabella", teacher, None,
                                      llm_request=lambda p: "ignore")
  assert out is None and "maria" not in strategy._STORE
  # offline llm stub: learner adopts
  out2 = transmission.run_transmission("maria", "isabella", teacher, None,
                                       llm_request=lambda p: "adopt")
  assert out2 is not None and out2["internal_copied_from_agent"] == "isabella"
  strategy._STORE.clear()


def main():
  tests = [test_legal_plan_passes, test_time_conflict, test_time_inverted,
           test_location_closed, test_resource_missing, test_unknown_location,
           test_no_path, test_world_rule_unknown_object, test_unmet_precondition,
           test_facility_unavailable, test_collects_multiple_failures,
           test_unconstrained_dimensions_pass,
           test_action_schema_fields, test_action_ids_unique,
           test_trade_success_changes_post_hash, test_trade_reject_is_verifiable_noop,
           test_state_hash_deterministic, test_no_economy_does_not_crash,
           test_record_ga_action_is_ledger_noop,
           test_classify_economic, test_classify_social_semantic_episodic,
           test_extract_resources, test_annotate_schema_and_confidence,
           test_rerank_identity_when_economy_off, test_rerank_boosts_economic_and_urgency,
           test_rerank_goal_resource_overlap, test_contradiction_filter_drops_lower_confidence,
           test_contradiction_filter_noop_single_polarity, test_apply_rerank_combined,
           test_condition_presets, test_condition_c3_adds_only_strategy,
           test_c2_and_c1_scaffolding, test_explicit_flag_overrides_preset,
           test_no_condition_all_off,
           test_strategy_schema_firewall_naming, test_strategy_ids_unique,
           test_strategy_lifecycle_transitions, test_strategy_evaluate_appends_fitness,
           test_strategy_mutate_links_parent, test_strategy_selection_summary_factual,
           test_parse_trade_json, test_parse_trade_lines, test_parse_trade_none_and_chatty,
           test_parse_trade_rejects_illegal, test_execute_decisions_changes_state,
           test_parse_strategy_proposal, test_strategy_store_and_active,
           test_strategy_proposal_prompt_factual, test_strategy_update_fitness_internal,
           test_parse_transmission_decision, test_transmission_adopt_records_lineage,
           test_transmission_adapt_is_mutation, test_transmission_run_ignore_and_adopt_offline]
  for t in tests:
    t()
    print(f"  PASS  {t.__name__}")
  print(f"\nALL {len(tests)} VALIDATOR SELF-TESTS PASS")


if __name__ == "__main__":
  main()
