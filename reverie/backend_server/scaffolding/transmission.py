"""Social transmission of strategies for C4 (guide 11.5 / 3.4).

C4 = C3 + this layer, and ONLY this (single-step). When agents converse, a
strategy can pass from teacher -> learner: the learner may adopt it as-is,
adapt it (mutation), or ignore it. The adopted strategy records
internal_copied_from_agent + internal_adoption_time; post-adoption fitness /
mutation reuse C3 (no new mechanism beyond "strategies can travel between
agents").

Firewall (2.6 / 7.7): every field here is agent self-reported -> agent_internal
(transmission + strategy channels). The post-hoc pipeline reconstructs
transmission/diffusion from raw/dialogue + raw/actions, and never reads
internal_copied_from_agent. The observable trace of teaching is the conversation
itself (already in raw/dialogue); raw/actions strategy_id stays null.

Deterministic core (parse + adopt) is offline-testable; the adopt/adapt/ignore
LLM decision is the live half. Terse structured output (econ_decision lesson).
"""
import re

from scaffolding import strategy as _strat


def _log(record):
  try:
    import telemetry_log
    telemetry_log.log_event("transmission", record)   # -> agent_internal
  except Exception:
    pass


_INSTRUCTION = (
    "{learner} hears {teacher} describe a strategy:\n"
    "  \"{summary}\"\n"
    "Decide what {learner} does with it. Output EXACTLY one line, no other text:\n"
    "adopt        (use it as-is)\n"
    "adapt: <one sentence revised plan>   (use a changed version)\n"
    "ignore       (do not use it)")

_RE_ADAPT = re.compile(r"adapt\s*:\s*(.+)", re.IGNORECASE)
_CHATTY = ("here", "sure", "okay", "let me", "i will", "as an", "based on",
           "would you", "```", "note:")


def build_transmission_prompt(learner_name, teacher_name, teacher_strategy):
  """Facts-only adoption-decision prompt (no steering): expose the teacher's
  strategy, ask adopt/adapt/ignore."""
  name = teacher_strategy.get("name", "a strategy")
  desc = teacher_strategy.get("description", "")
  summary = f"{name}: {desc}".strip().strip(":")
  return _INSTRUCTION.format(learner=learner_name, teacher=teacher_name,
                             summary=summary)


def parse_transmission_decision(raw):
  """Parse the learner's reply -> ('adopt', None) | ('adapt', new_desc) |
  ('ignore', None) | None (garbled). Robust to chatty prose. Never raises."""
  if not raw or not str(raw).strip():
    return None
  for line in str(raw).strip().splitlines():
    line = line.strip(" \t-*•")
    low = line.lower()
    if not line:
      continue
    m = _RE_ADAPT.search(line)
    if m and len(m.group(1).strip()) >= 5:
      return ("adapt", m.group(1).strip().strip('"'))
    if low.startswith("adopt"):
      return ("adopt", None)
    if low.startswith("ignore"):
      return ("ignore", None)
    if any(c in low for c in _CHATTY):
      continue
  return None


def adopt(learner_name, teacher_name, teacher_strategy, adoption_time,
          *, adapt_desc=None):
  """Create the adopted strategy in the learner's store (status trial), tagged
  with internal_copied_from_agent + internal_adoption_time + parent = teacher's
  strategy id. `adapt_desc` (mutation) records a note + uses the revised plan.
  Logs to agent_internal. Returns the new strategy."""
  name = teacher_strategy.get("name", "adopted strategy")
  desc = adapt_desc or teacher_strategy.get("description", "")
  child = _strat.new_strategy(
      learner_name, name, desc,
      required_resources=teacher_strategy.get("required_resources"),
      parent_id=teacher_strategy.get("internal_strategy_id"),
      created_at=adoption_time)
  child["status"] = "trial"
  child["internal_copied_from_agent"] = teacher_name
  child["internal_adoption_time"] = adoption_time
  if adapt_desc:
    child["internal_mutations_self_reported"].append(f"adapted: {adapt_desc}")
  _strat.record_strategy(learner_name, child,
                         event=("adapted" if adapt_desc else "adopted"))
  _log({"persona": learner_name, "event": ("adapt" if adapt_desc else "adopt"),
        "internal_copied_from_agent": teacher_name,
        "internal_adoption_time": adoption_time,
        "internal_strategy_id": child["internal_strategy_id"],
        "internal_self_reported_parent": teacher_strategy.get("internal_strategy_id")})
  return child


def run_transmission(learner_name, teacher_name, teacher_strategy, curr_time,
                     *, llm_request=None):
  """One adoption decision (1 LLM call). Shows the learner the teacher's
  strategy, parses adopt/adapt/ignore, records the outcome. Returns the adopted
  strategy or None. Guarded; never raises."""
  try:
    if not teacher_strategy:
      return None
    try:
      adoption_time = curr_time.strftime("%A %B %d")
    except Exception:
      adoption_time = ""
    prompt = build_transmission_prompt(learner_name, teacher_name, teacher_strategy)
    if llm_request is None:
      try:
        from persona.prompt_template.gpt_structure import ChatGPT_single_request
        llm_request = ChatGPT_single_request
      except Exception:
        return None
    try:
      raw = llm_request(prompt)
    except Exception:
      raw = None
    decision = parse_transmission_decision(raw)
    _log({"persona": learner_name, "event": "adoption_attempt",
          "teacher": teacher_name,
          "teacher_strategy_id": teacher_strategy.get("internal_strategy_id"),
          "raw_response": (str(raw)[:400] if raw is not None else None),
          "decision": (decision[0] if decision else None)})
    if not decision or decision[0] == "ignore":
      return None
    return adopt(learner_name, teacher_name, teacher_strategy, adoption_time,
                 adapt_desc=(decision[1] if decision[0] == "adapt" else None))
  except Exception:
    return None
