"""Typed memory annotation for C2.5 (guide 7.3).

Design choice (faithfulness + low risk): instead of changing GA's ConceptNode
constructor / nodes.json save-load (which would force a memory-schema migration
and risk diverging from the original GA), we derive the typed view ON DEMAND
from a node's existing fields with deterministic rules. GA memory behavior is
therefore byte-for-byte unchanged; the typed annotation is a read-only overlay
that downstream retrieval rerank / contradiction filter (step 4) consume, and
that gets logged to agent_internal/ (internal representation, firewall-forbidden
to evaluators).

mem_type enum (guide 7.3): episodic / semantic / social / economic / procedural
/ strategic / invalidated. C2.5 produces episodic/semantic/social/economic;
`strategic` is reserved (it is a by-product of the C3 strategy object) and never
emitted here; `invalidated` is set by the contradiction filter, not at creation.

Type tagging is RULE-BASED (no extra LLM call -> no added API cost, OQ-2 default).
`confidence` is a deterministic proxy from poignancy (NOT a Bayesian posterior);
`money_delta` is not recoverable from free text here, so it stays 0 (a real value
only exists for economy-engine trade actions, captured in the typed action API).
"""
import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
  sys.path.insert(0, _BACKEND)

try:
  from economy import config as _econ_config
  # Include phase-scheduled resources such as tool, not only day-0 resources.
  _RESOURCES = [r.lower() for r in _econ_config.PRICES]
except Exception:
  _RESOURCES = []

_MONEY_WORDS = {
    "cash", "money", "buy", "buys", "buying", "bought", "sell", "sells",
    "selling", "sold", "sale", "pay", "pays", "paid", "paying", "price",
    "prices", "cost", "costs", "income", "salary", "wage", "wages", "trade",
    "trading", "market", "customer", "customers", "profit", "earn", "earned",
    "earnings", "rent", "afford", "broke", "bankrupt",
}

_SOCIAL_WORDS = {
    "talk", "talks", "talking", "talked", "chat", "chatting", "chatted", "met",
    "meeting", "meet", "conversation", "discuss", "discussing", "greet",
    "greeting", "greeted", "with", "friend", "friends", "together",
}


def _tokens(node):
  text = (getattr(node, "description", "") or "").lower()
  toks = set(text.replace(".", " ").replace(",", " ").split())
  try:
    toks |= {str(k).lower() for k in (getattr(node, "keywords", None) or [])}
  except Exception:
    pass
  return text, toks


def extract_resources(node):
  """Economy resources mentioned by this node (deterministic, ordered)."""
  text, _ = _tokens(node)
  return [r for r in _RESOURCES if r in text]


def _is_economic(text, toks, resources):
  return bool(resources) or bool(toks & _MONEY_WORDS)


def _is_social(text, toks):
  return bool(toks & _SOCIAL_WORDS)


def classify(node):
  """Return the mem_type for a ConceptNode. Deterministic; order matters:
  economic content wins, then chat=social, then thought=semantic, then social
  verbs, else episodic."""
  node_type = getattr(node, "type", "event")
  text, toks = _tokens(node)
  resources = [r for r in _RESOURCES if r in text]
  if _is_economic(text, toks, resources):
    return "economic"
  if node_type == "chat":
    return "social"
  if node_type == "thought":
    return "semantic"
  if _is_social(text, toks):
    return "social"
  return "episodic"


def _confidence(node):
  # poignancy is GA's 1-10 importance; use it as a deterministic confidence
  # PROXY (clamped to [0.1, 1.0]). Not a real posterior -- documented as such.
  p = getattr(node, "poignancy", 5)
  try:
    p = float(5 if p is None else p)
  except Exception:
    p = 5.0
  return round(min(1.0, max(0.1, p / 10.0)), 2)


def annotate(node):
  """Read-only typed-memory overlay for a ConceptNode (guide 7.3 schema subset).
  Never mutates the node; never raises."""
  try:
    resources = extract_resources(node)
    return {
        "memory_id": getattr(node, "node_id", None),
        "mem_type": classify(node),
        "summary": getattr(node, "description", None),
        "resources": resources,
        "money_delta": 0,                 # not recoverable from free text
        "confidence": _confidence(node),
        "source": getattr(node, "type", "event"),
        "evidence_ids": list(getattr(node, "filling", None) or []),
        "valid_until": None,
    }
  except Exception:
    return {"memory_id": getattr(node, "node_id", None), "mem_type": "episodic",
            "resources": [], "money_delta": 0, "confidence": 0.5,
            "source": "event", "evidence_ids": [], "valid_until": None}
