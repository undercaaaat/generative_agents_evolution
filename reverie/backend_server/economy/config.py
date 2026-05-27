"""Default economic parameters for the minimal P3 economy.

All values are FIXED in P3 phase-1 (basic survival, guide 6.1): fixed wages,
fixed prices, fixed daily cost. Price fluctuation / shocks come later (P7).
Everything here is deterministic -- no RNG -- so runs are reproducible.

These defaults are the single source the spec (02-protocols/environment-spec.md)
documents. Tuning goal (P3 acceptance): a working agent stays solvent; a
non-working agent depletes gradually (pressure exists, no instant collapse).
"""
import os

# --- per-agent starting condition ---
INITIAL_CASH = 100.0          # currency units
DAILY_NEED_COST = 20.0        # deducted once per game day (food/housing/etc.)

# --- safety net (guide 5.5: avoid premature total collapse) ---
EMERGENCY_RELIEF = 5.0        # daily relief credited when cash <= RELIEF_THRESHOLD
RELIEF_THRESHOLD = 0.0        # cash at/below this triggers relief (and bankruptcy flag)

# --- resources & fixed market prices (3-5 resources, guide 5.2) ---
RESOURCES = ["food", "coffee", "ingredients"]
PRICES = {                    # base market unit price (buy == sell, no spread)
    "food": 5.0,
    "coffee": 3.0,
    "ingredients": 2.0,
}

# --- minimal price dynamics (economic-action-loop.md 6; pulled forward from P7) ---
# Deterministic triangle wave around the base price, indexed by day. NO RNG ->
# fully reproducible. AMPLITUDE defaults to 0.0 = FLAT (identical to P3 fixed
# price, preserves determinism + existing tests); the capability pilot sets it
# >0 to open buy-low/sell-high windows ACROSS days (same-day round-trip still
# nets zero -> no risk-free arbitrage; cross-day profit needs holding + timing,
# i.e. a real strategy). Final value is tuned + frozen at P10.
# Env override lets the capability pilot enable dynamics without editing the
# (to-be-frozen) default. Unset -> 0.0 (flat, P3 behavior).
PRICE_WAVE_AMPLITUDE = float(os.environ.get("GA_PRICE_WAVE_AMPLITUDE", "0.0") or "0.0")
PRICE_WAVE_PERIOD = 4         # days per full cycle
PRICE_WAVE_PHASE = {          # per-resource phase offset (days) so they desync
    "food": 0,
    "coffee": 1,
    "ingredients": 2,
}

# --- jobs (income, guide 5.3 first tier: fixed/temp work) ---
# wage_per_hour is credited per step pro-rata (wage_per_hour * step_seconds/3600)
# while the agent is "working": within work_hours AND its activity/location
# matches one of the keyword sets. Tuned so ~10 work-hours/day > DAILY_NEED_COST.
DEFAULT_WAGE_PER_HOUR = 4.0
JOBS = {
    "Isabella Rodriguez": {
        "wage_per_hour": 4.0,
        "work_hours": (8, 20),
        "location_keywords": ["Hobbs Cafe", "cafe"],
        "activity_keywords": ["counter", "serving", "cafe", "coffee", "working"],
    },
    "Maria Lopez": {
        "wage_per_hour": 4.0,
        "work_hours": (10, 23),
        "location_keywords": [],
        "activity_keywords": ["stream", "twitch", "gaming", "streaming"],
    },
    "Klaus Mueller": {
        "wage_per_hour": 4.0,
        "work_hours": (8, 18),
        "location_keywords": ["library", "Oak Hill College"],
        "activity_keywords": ["writing", "research", "reading", "library"],
    },
}


# De-semanticized transfer test (env-spec 0c): optionally decouple wage income
# from semantic venues. Default off. When GA_NEUTRAL_JOBS=1, every persona gets
# the same generic job whose keywords are NOT venue/good names ("Hobbs Cafe",
# "coffee"), so income still accrues when the agent's plan says it is "working"
# but is not tied to an occupational prior. Note: under interpretation A job
# keywords are matched against the agent's act_description (NOT injected into any
# prompt), so this does not affect prompt leakage -- it only decouples
# occupation from income for the transfer pilot. Keywords kept non-empty to
# avoid a floor effect (agents must still be able to earn).
_GENERIC_JOB = {
    "wage_per_hour": DEFAULT_WAGE_PER_HOUR,
    "work_hours": (9, 18),
    "location_keywords": [],
    "activity_keywords": ["working", "work"],
}


def neutral_jobs_on():
  return os.environ.get("GA_NEUTRAL_JOBS", "0") not in ("0", "false", "False", "")


def job_for(persona_name):
  """Return the job config for a persona, or a generic default. Under
  GA_NEUTRAL_JOBS the generic (venue-free) job is used for everyone."""
  if neutral_jobs_on():
    return dict(_GENERIC_JOB)
  return JOBS.get(persona_name, dict(_GENERIC_JOB))
