"""Neutral-persona override for the de-semanticized transfer test (P7, P11).

Prior-knowledge confound (environment-spec 0c): if a persona starts the
de-semanticized test already "knowing" it runs a cafe, any observed economic
competence is contaminated by an occupational prior rather than learned
structure. This module strips the OCCUPATIONAL / economic backstory from a
persona's identity text at load time, keeping everything else (name, the map,
spatial memory) intact -- the "interpretation A" decision (2026-05-27): only
goods/signals + persona occupation must be abstract; venue names in the map are
acceptable and the economic decision surface does not depend on them.

Gated by GA_NEUTRAL_PERSONAS (default off -> personas unchanged, baseline
faithful). Applied right after each Persona is constructed in reverie.

What it rewrites (the identity-stable-set text fed to prompts via get_str_iss):
  learned / currently / daily_plan_req / lifestyle  -> neutral, name-templated.
What it deliberately leaves alone:
  name, first_name, innate (personality traits, not occupational), and ALL
  spatial memory / map (interpretation A). The smoke test (P11 runbook) dumps
  real prompts to confirm no occupational prior leaks through residual fields.
"""
import os


def neutral_personas_on():
  return os.environ.get("GA_NEUTRAL_PERSONAS", "0") not in ("0", "false", "False", "")


# Occupation-bearing identity fields and their neutral, name-templated text.
def _neutral_text(field, name):
  return {
      "learned": f"{name} is a resident of the area and keeps a regular daily routine.",
      "currently": f"{name} is going about an ordinary day.",
      "daily_plan_req": (f"{name} keeps a regular routine during the day and "
                         f"has no fixed occupation."),
      "lifestyle": f"{name} wakes in the morning and sleeps at night.",
  }.get(field, None)


_FIELDS = ("learned", "currently", "daily_plan_req", "lifestyle")


def neutralize_scratch(scratch, name):
  """Rewrite occupation-bearing identity fields on a scratch-like object to
  neutral text. Returns the list of fields actually changed. Pure w.r.t. the
  map / spatial memory (untouched). Never raises."""
  changed = []
  for f in _FIELDS:
    txt = _neutral_text(f, name)
    if txt is not None and hasattr(scratch, f):
      try:
        setattr(scratch, f, txt)
        changed.append(f)
      except Exception:
        pass
  return changed


def neutralize_persona(persona):
  """Neutralize a GA Persona in place (no-op-safe). Returns changed fields."""
  try:
    scratch = getattr(persona, "scratch", None)
    if scratch is None:
      return []
    name = getattr(scratch, "name", None) or getattr(persona, "name", "Agent")
    return neutralize_scratch(scratch, name)
  except Exception:
    return []


def _selftest():
  ok = True

  class _FakeScratch:
    def __init__(self):
      self.name = "Isabella Rodriguez"
      self.learned = ("Isabella Rodriguez is a cafe owner of Hobbs Cafe who "
                      "loves to make people feel welcome.")
      self.currently = "Isabella is excited for the Valentine's party at Hobbs Cafe."
      self.daily_plan_req = "Wake at 6am, open Hobbs Cafe by 7:30am, serve coffee."
      self.lifestyle = "Isabella runs the cafe from morning to evening."
      self.innate = "warm, social, organized"

  s = _FakeScratch()
  changed = neutralize_scratch(s, s.name)
  # Economic/occupational markers that must be gone after neutralization.
  markers = ("cafe", "coffee", "hobbs", "serve", "owner", "open ")
  blob = " ".join([s.learned, s.currently, s.daily_plan_req, s.lifestyle]).lower()
  leaks = [m for m in markers if m in blob]
  if leaks:
    print(f"FAIL: occupational markers survived neutralization: {leaks}"); ok = False
  if set(changed) != set(_FIELDS):
    print(f"FAIL: expected to change {_FIELDS}, changed {changed}"); ok = False
  if s.innate != "warm, social, organized":
    print("FAIL: innate (personality) should be untouched"); ok = False
  if s.name != "Isabella Rodriguez":
    print("FAIL: name should be kept (interpretation A)"); ok = False
  print("neutral_persona selftest: ALL PASS" if ok
        else "neutral_persona selftest: FAILED")
  return ok


if __name__ == "__main__":
  import sys
  sys.exit(0 if _selftest() else 1)
