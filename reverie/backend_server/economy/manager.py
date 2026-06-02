"""EconomyManager: deterministic background settlement for the P3 economy.

Lifecycle per simulation step (called from reverie's main loop when enabled):
  manager.step(personas, curr_time, step_seconds)
    1. ensure each agent has an economic state;
    2. accrue job income for agents that are "working" this step (pro-rata);
    3. on a new game day, deduct DAILY_NEED_COST, apply safety-net relief,
       and set/clear the bankruptcy flag;
    4. log each agent's economic snapshot to telemetry raw/economy.jsonl.

Agents do NOT perceive any of this in P3 (option A): it is a shadow ledger
driven by whatever the GA loop already decided the agent does. The trade()
primitive exists and is unit-tested, but is not invoked by the loop until
economic perception is added (C2+).

Hard rule: like telemetry, this must never raise into the simulation. step()
swallows its own exceptions; settlement math is plain and deterministic.
"""
import os

from economy import config

try:
  import telemetry_log
except Exception:
  telemetry_log = None

try:
  from economy import signals
except Exception:
  signals = None


def economy_enabled():
  return os.environ.get("GA_ECONOMY", "0") not in ("0", "false", "False", "")


# --- Environment model registry (P7 signal/causal layer) -------------------
# When an EnvironmentModel is attached, the manager sources its resources +
# base prices and applies the signal-driven price multiplier + inventory caps
# from it (the causal layer supersedes the P3 placeholder triangle wave).
# Default None -> pure P3 behavior (fixed PRICES + triangle wave), so every
# existing economy test is unchanged. The model is fully frozen at t=0 (guide
# 6.6 / 11.0): no RNG, no runtime researcher input.
_ENV_MODEL = None


def set_env_model(model):
  """Attach (or clear with None) the active EnvironmentModel."""
  global _ENV_MODEL
  _ENV_MODEL = model


def get_env_model():
  return _ENV_MODEL


def load_env_config(path=None):
  """Load + attach an EnvironmentModel from `path` or $GA_ENV_CONFIG. Returns
  the model, or None if unset/unavailable (then P3 behavior holds)."""
  path = path or os.environ.get("GA_ENV_CONFIG")
  if not path or signals is None:
    return None
  try:
    model = signals.load_env(path)
    set_env_model(model)
    return model
  except Exception:
    return None


# --- Shock schedule registry (P7) ------------------------------------------
# A registered ShockSchedule applies a deterministic price multiplier and
# resource availability (store_closure blocks buys) by day index. Default None
# -> no shocks (P3 behavior). Frozen at t=0; no RNG, no runtime intervention.
_SHOCKS = None


def set_shock_schedule(sched):
  global _SHOCKS
  _SHOCKS = sched


def get_shock_schedule():
  return _SHOCKS


# --- Phase schedule registry (P13 main-run orchestration) ------------------
# Default None preserves the pilot/P3 behavior. When attached, the manager
# applies the t=0-frozen phase parameters by day: active resources, whether env
# signals are active, and the fallback triangle-wave amplitude.
_PHASES = None


def set_phase_schedule(sched):
  global _PHASES
  _PHASES = sched


def get_phase_schedule():
  return _PHASES


# --- Economic perception registry (C2) -------------------------------------
# Module-level so prompt builders (run_gpt_prompt) can read the current
# perception string by persona name without holding the manager instance,
# mirroring the telemetry_log pattern. EconomyManager.stamp_perception()
# refreshes this each step BEFORE personas plan.
_PERCEPTION = {}


def current_perception(name):
  """Return the latest economic-perception string for a persona, or None.
  Used by C2 prompt injection (daily plan). None when economy is off."""
  return _PERCEPTION.get(name)


def _set_perception(name, text):
  _PERCEPTION[name] = text


# Economic-context registry (C2.5 retrieval rerank). Numeric counterpart of the
# perception string: lets scaffolding/retrieval reweight memories by the agent's
# current economic state without holding the manager instance. None when economy
# is off -> rerank stays identity (GA/C2 behavior unchanged).
_ECON_CTX = {}


def current_econ_context(name):
  """Return {'cash','daily_cost','bankrupt'} for a persona, or None if unset."""
  return _ECON_CTX.get(name)


def _set_econ_context(name, ctx):
  _ECON_CTX[name] = ctx


class EconomyManager:
  def __init__(self, cfg=None):
    self.cfg = cfg or config
    self.states = {}          # persona_name -> {cash, inventory, bankrupt, ...}
    self._last_day = None     # date of the last settled day (for rollover)
    self._day0 = None         # first date seen (day index origin for price wave)
    self._curr_day_index = 0  # days since _day0 (drives deterministic price wave)

  # -- price dynamics (deterministic, reproducible) ------------------------
  def _set_day(self, curr_time):
    """Record the current day index from curr_time (origin = first day seen).
    Safe with curr_time None (leaves index unchanged)."""
    try:
      d = curr_time.date()
      if self._day0 is None:
        self._day0 = d
      self._curr_day_index = (d - self._day0).days
    except Exception:
      pass

  def _resources(self, day_index=None):
    """Active resource list: phase override, then env model, then config."""
    phase = self.active_phase(day_index)
    if phase and phase.get("active_resources"):
      return list(phase["active_resources"])
    m = _ENV_MODEL
    return m.resources if (m is not None and m.resources) else self.cfg.RESOURCES

  def active_phase(self, day_index=None):
    """Return the attached frozen phase for a day, or None when phase-off."""
    di = self._curr_day_index if day_index is None else day_index
    return _PHASES.phase_for(di) if _PHASES is not None else None

  def wage_multiplier(self, day_index=None):
    """Wage scaling from a registered wage_cut shock on a given day (default
    1.0 with no shocks attached -> byte-for-byte P3 wage behavior)."""
    di = self._curr_day_index if day_index is None else day_index
    return _SHOCKS.wage_multiplier(di) if _SHOCKS is not None else 1.0

  def price(self, resource, day_index=None):
    """Market unit price of a resource on a given day. Layers (all deterministic):
      1. base: env-model base price if attached, else config PRICES;
      2. dynamics: env-model signal/causal multiplier if attached, else the P3
         triangle wave (amplitude 0 -> flat);
      3. shock: registered ShockSchedule price multiplier (price_spike /
         demand_shift), default none.
    Default (no model, no shocks) is byte-for-byte P3 behavior."""
    di = self._curr_day_index if day_index is None else day_index
    phase_cfg = self.active_phase(di)
    if resource not in self._resources(di):
      return None
    m = _ENV_MODEL
    if m is not None:
      base = m.base_price(resource)
      if base is None:
        base = self.cfg.PRICES.get(resource)
      if base is None:
        return None
      signals_active = (phase_cfg.get("signals_active", True)
                        if phase_cfg is not None else True)
      p = base * (m.price_multiplier(resource, di) if signals_active else 1.0)
    else:
      base = self.cfg.PRICES.get(resource)
      if base is None:
        return None
      amp = (phase_cfg.get("price_wave_amplitude", 0.0)
             if phase_cfg is not None
             else getattr(self.cfg, "PRICE_WAVE_AMPLITUDE", 0.0)) or 0.0
      if amp == 0.0:
        p = base
      else:
        period = getattr(self.cfg, "PRICE_WAVE_PERIOD", 4) or 4
        phase = getattr(self.cfg, "PRICE_WAVE_PHASE", {}).get(resource, 0)
        pos = ((di + phase) % period) / float(period)   # [0,1)
        tri = 1.0 - 4.0 * abs(pos - 0.5)                 # triangle in [-1, 1]
        p = base * (1.0 + amp * tri)
    if _SHOCKS is not None:
      p *= _SHOCKS.price_multiplier(resource, di)
    return round(p, 2)

  # -- state ---------------------------------------------------------------
  def ensure_agent(self, name):
    if name not in self.states:
      self.states[name] = {
          "cash": float(self.cfg.INITIAL_CASH),
          "inventory": {r: 0 for r in self._resources()},
          "bankrupt": False,
          "income_today": 0.0,
          "days_bankrupt": 0,
      }
    else:
      for resource in self._resources():
        self.states[name]["inventory"].setdefault(resource, 0)
    return self.states[name]

  def snapshot(self, name):
    s = self.ensure_agent(name)
    return {
        "persona": name,
        "cash": round(s["cash"], 4),
        "inventory": dict(s["inventory"]),
        "bankrupt": s["bankrupt"],
        "income_today": round(s["income_today"], 4),
    }

  # -- income --------------------------------------------------------------
  def _is_working(self, name, act_description, act_address, hour):
    job = self.cfg.job_for(name)
    lo, hi = job["work_hours"]
    if not (lo <= hour < hi):
      return False
    text = f"{act_description or ''} {act_address or ''}".lower()
    for kw in job["location_keywords"] + job["activity_keywords"]:
      if kw.lower() in text:
        return True
    return False

  def accrue_income(self, name, act_description, act_address, hour, step_seconds):
    s = self.ensure_agent(name)
    if s["bankrupt"]:
      pass  # bankrupt agents can still earn their way back out
    if self._is_working(name, act_description, act_address, hour):
      wage = self.cfg.job_for(name)["wage_per_hour"] * self.wage_multiplier()
      earned = wage * (float(step_seconds) / 3600.0)
      s["cash"] += earned
      s["income_today"] += earned

  # -- daily settlement ----------------------------------------------------
  def _settle_day(self, name):
    s = self.ensure_agent(name)
    s["cash"] -= self.cfg.DAILY_NEED_COST
    if s["cash"] <= self.cfg.RELIEF_THRESHOLD:
      s["cash"] += self.cfg.EMERGENCY_RELIEF   # safety net
      s["bankrupt"] = True
      s["days_bankrupt"] += 1
    else:
      s["bankrupt"] = False
    s["income_today"] = 0.0

  # -- transaction primitive (unit-tested; not loop-driven until C2) -------
  def trade(self, name, resource, qty, side="buy"):
    """Buy/sell `qty` of `resource` at the fixed market price. Returns True on
    success, False if invalid (unknown resource, insufficient cash/inventory)."""
    s = self.ensure_agent(name)
    unit = self.price(resource)
    if unit is None or qty <= 0:
      return False
    cost = unit * qty
    if side == "buy":
      if s["cash"] < cost:
        return False
      # Store-closure shock (P7): the resource cannot be bought while closed
      # (selling still allowed). Default (no shocks) -> always available.
      if _SHOCKS is not None and not _SHOCKS.is_available(resource, self._curr_day_index):
        return False
      # Inventory cap (P7 inventory_cap causal effect): reject buys that would
      # exceed the cap active for this resource today. None -> uncapped (P3).
      m = _ENV_MODEL
      if m is not None:
        cap = m.inventory_cap(resource, self._curr_day_index)
        if cap is not None and s["inventory"].get(resource, 0) + qty > cap:
          return False
      s["cash"] -= cost
      s["inventory"][resource] = s["inventory"].get(resource, 0) + qty
    elif side == "sell":
      if s["inventory"].get(resource, 0) < qty:
        return False
      s["inventory"][resource] -= qty
      s["cash"] += cost
    else:
      return False
    return True

  # -- economic perception (C2) --------------------------------------------
  def perception_str(self, name, date_str=""):
    """Compact, factual economic-context block for prompt injection. Only
    objective facts -- NO advice/imperatives (guide 13.4: avoid steering)."""
    s = self.ensure_agent(name)
    # Roundtable prompts call perception_str() directly rather than going
    # through stamp_perception(). Keep the retrieval reranker context current
    # on both paths so C2.5 sees the same factual economy snapshot.
    _set_econ_context(name, {"cash": round(s["cash"], 2),
                             "daily_cost": float(self.cfg.DAILY_NEED_COST),
                             "bankrupt": s["bankrupt"]})
    prices = ", ".join(f"{r} {self.price(r)}" for r in self._resources())
    inv = ", ".join(f"{r} x{s['inventory'].get(r, 0)}" for r in self._resources())
    status = "bankrupt" if s["bankrupt"] else "solvent"
    header = (f"[Economic situation as of {date_str}]" if date_str
              else "[Economic situation]")
    return "\n".join([
        header,
        (f"Cash: {round(s['cash'], 1)}. Daily living cost: "
         f"{self.cfg.DAILY_NEED_COST} per day. Status: {status}."),
        f"Inventory: {inv}.",
        f"Market prices: {prices}.",
        f"Income earned today so far: {round(s['income_today'], 1)}.",
    ])

  def stamp_perception(self, personas, curr_time):
    """Refresh the perception registry for all personas. Call BEFORE move()
    so daily-plan prompts can read it. Never raises."""
    try:
      self._set_day(curr_time)
      for name, persona in personas.items():
        self.ensure_agent(name)
        try:
          date_str = persona.scratch.get_str_curr_date_str()
        except Exception:
          date_str = str(curr_time)
        _set_perception(name, self.perception_str(name, date_str))
        s = self.ensure_agent(name)
        _set_econ_context(name, {"cash": round(s["cash"], 2),
                                 "daily_cost": float(self.cfg.DAILY_NEED_COST),
                                 "bankrupt": s["bankrupt"]})
    except Exception:
      pass

  # -- main loop entry -----------------------------------------------------
  def step(self, personas, curr_time, step_seconds):
    """Settle one simulation step for all personas. Never raises."""
    try:
      hour = curr_time.hour
      day = curr_time.date()
      self._set_day(curr_time)
      rolled_over = self._last_day is not None and day != self._last_day
      for name, persona in personas.items():
        self.ensure_agent(name)
        if rolled_over:
          self._settle_day(name)
        act_desc = getattr(persona.scratch, "act_description", None)
        act_addr = getattr(persona.scratch, "act_address", None)
        self.accrue_income(name, act_desc, act_addr, hour, step_seconds)
        if telemetry_log is not None:
          telemetry_log.log_event("economy", self.snapshot(name))
      self._last_day = day
    except Exception:
      pass
