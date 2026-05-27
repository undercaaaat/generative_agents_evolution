"""Constraint planner / validator for C2.5 (guide 7.8).

An LLM may *propose* a plan, but it cannot directly execute it: every plan must
pass this validator first. The validator is a pure, deterministic function over
an explicit plan dict + world context (`PlanContext`) -- it does NOT reach into
live persona objects, so it is fully offline-testable (see scaffolding/selftest.py)
and seed-reproducible. Live wiring (building a PlanContext from real personas /
maze / economy and hooking it into plan.py) is a separate, later step.

8 checks from guide 7.8:
  1 time conflict        2 location open        3 resources exist
  4 knows location       5 path exists          6 contract  (n/a first paper)
  7 stamina (n/a)        8 world rules

Contract and stamina have no backing state variable in the first paper's
economy, so they always pass; this is recorded explicitly (environment-spec)
rather than silently dropped, so the validator's surface matches guide 7.8 and
can be tightened when those layers arrive (second paper).

Telemetry: the plan object and validator_result are internal cognition ->
agent_internal/plans. A rejected *action* is an observable behavioral
constraint and may additionally be logged to raw/actions, symmetrically across
all typed conditions (C2.5/C3/C4). This module does not log; callers do.
"""
import os
import sys

# Make backend_server/ importable whether run as a module or a script, so we can
# ground the world-rule check in the actual economy resource set.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

try:
  from economy import config as _econ_config
  _ECON_RESOURCES = set(_econ_config.RESOURCES)
except Exception:  # economy optional / import guarded -- never break validation
  _ECON_RESOURCES = set()


class PlanContext:
  """Everything the validator needs to know about the world for one plan.

  Defaults are permissive (a well-formed plan passes); callers populate the
  fields they can observe. `None` for a set field means "unconstrained / not
  modeled" (that check passes). This keeps the validator honest: we only fail a
  plan on a dimension we actually have ground truth for.
  """

  def __init__(self,
               busy_intervals=None,        # list[(start_min, end_min)] already-committed time
               known_locations=None,       # set[str] locations the agent knows (spatial memory); None = knows all
               open_locations=None,        # set[str] currently-open locations;        None = all open
               reachable_locations=None,   # set[str] locations with a path;           None = all reachable
               inventory=None,             # dict[str,int] economic goods on hand
               facilities=None,            # set[str] facility tokens that exist in the world
               available_facilities=None,  # set[str] facilities currently usable;      None = all available
               satisfied_preconditions=None):  # set[str] precondition flags currently true
    self.busy_intervals = list(busy_intervals or [])
    self.known_locations = known_locations
    self.open_locations = open_locations
    self.reachable_locations = reachable_locations
    self.inventory = dict(inventory or {})
    self.facilities = set(facilities or [])
    self.available_facilities = available_facilities
    self.satisfied_preconditions = set(satisfied_preconditions or [])


class ValidationResult:
  def __init__(self, failures):
    self.failures = list(failures)  # list[(check_name, reason)]

  @property
  def ok(self):
    return not self.failures

  @property
  def result(self):
    return "pass" if self.ok else "fail"

  def reason(self):
    return "; ".join(f"{c}: {r}" for c, r in self.failures)

  def as_dict(self):
    return {"validator_result": self.result,
            "failures": [{"check": c, "reason": r} for c, r in self.failures]}

  def __repr__(self):
    return f"ValidationResult({self.result}: {self.reason() or 'ok'})"


def _overlap(a_start, a_end, b_start, b_end):
  return a_start < b_end and b_start < a_end


def _check_time(plan, ctx):
  s, e = plan.get("start_min"), plan.get("end_min")
  if s is None or e is None:
    return None  # untimed plan: nothing to check
  if e <= s:
    return "time", f"end_min ({e}) <= start_min ({s})"
  for bs, be in ctx.busy_intervals:
    if _overlap(s, e, bs, be):
      return "time", f"overlaps committed interval ({bs}-{be})"
  return None


def _check_location_open(plan, ctx):
  loc = plan.get("location")
  if loc is None or ctx.open_locations is None:
    return None
  if loc not in ctx.open_locations:
    return "location_open", f"{loc!r} not open"
  return None


def _check_knows_location(plan, ctx):
  loc = plan.get("location")
  if loc is None or ctx.known_locations is None:
    return None
  if loc not in ctx.known_locations:
    return "knows_location", f"agent does not know {loc!r}"
  return None


def _check_path(plan, ctx):
  loc = plan.get("location")
  if loc is None or ctx.reachable_locations is None:
    return None
  if loc not in ctx.reachable_locations:
    return "path", f"no path to {loc!r}"
  return None


def _check_resources(plan, ctx):
  # required_resources entries are resolved against three worlds, in order:
  #   economic good  -> must have inventory qty >= 1
  #   known facility -> must be currently available
  #   anything else  -> world-rule violation (object doesn't exist)
  for r in plan.get("required_resources", []):
    if r in _ECON_RESOURCES:
      if ctx.inventory.get(r, 0) < 1:
        return "resources", f"missing economic good {r!r}"
    elif r in ctx.facilities:
      if ctx.available_facilities is not None and r not in ctx.available_facilities:
        return "resources", f"facility {r!r} unavailable"
    else:
      return "world_rules", f"unknown world object {r!r}"
  return None


def _check_preconditions(plan, ctx):
  for pc in plan.get("preconditions", []):
    if pc not in ctx.satisfied_preconditions:
      return "world_rules", f"unmet precondition {pc!r}"
  return None


def _check_contract(plan, ctx):
  return None  # n/a first paper (no contract layer); placeholder passes


def _check_stamina(plan, ctx):
  return None  # n/a first paper (no stamina variable); placeholder passes


_CHECKS = (_check_time, _check_location_open, _check_knows_location, _check_path,
           _check_resources, _check_preconditions, _check_contract, _check_stamina)


def validate_plan(plan, ctx=None):
  """Run all 8 checks. Returns a ValidationResult (collects every failure, so a
  caller / log sees all problems at once, not just the first). Never raises."""
  ctx = ctx or PlanContext()
  failures = []
  for check in _CHECKS:
    try:
      out = check(plan, ctx)
    except Exception as ex:  # a buggy check must not crash the sim loop
      out = ("validator_error", f"{check.__name__}: {ex}")
    if out is not None:
      failures.append(out)
  return ValidationResult(failures)
