"""Evidence-aware retrieval rerank for C2.5 (guide 7.4).

Strictly +2 steps on top of GA's `recency + importance + relevance`:
    economic_relevance_rerank   -- weight memories by relevance to the agent's
                                   CURRENT economic state + current goal
    contradiction_filter        -- drop a candidate that contradicts a
                                   higher-confidence existing memory

NOT query decomposition / evidence compression / social rerank (guide 7.4 defers
those). Both steps are pure, deterministic functions over a {node_id: score}
dict plus an id->node map and an economic context; they are wired into
new_retrieve() after the GA master score, before top-x truncation (step 5).

Single-step-increment guard (-> P5 / OQ-1): in C2.5 the rerank keys off the
CURRENT economic state + current goal, NOT a strategy object (there is none).
The recommended C3 design keeps this formula UNCHANGED so that C3-vs-C2.5 stays
exactly the strategy object; strategy influence flows through planning/selection,
not by mutating this rerank.

Economy-off / C2: with econ_ctx=None the rerank is the IDENTITY and the filter
is a no-op, so GA/C2 retrieval is byte-for-byte unchanged.

These steps are internal cognition -> agent_internal/retrieve (firewall). They
never feed an evaluator metric, so the contradiction heuristic's coarseness
(OQ-5) cannot threaten main results.
"""
from scaffolding import typed_memory

# Coarse economic polarity lexicon for the contradiction heuristic (OQ-5:
# parameterizable; internal-only). "Contradiction" = two economic memories on
# the same resource topic with opposite polarity.
_POS_WORDS = {"profit", "profitable", "earned", "earn", "income", "sold", "sell",
              "selling", "gain", "gains", "demand", "popular", "busy", "growth"}
_NEG_WORDS = {"loss", "lost", "cost", "costly", "expensive", "broke", "bankrupt",
              "unsold", "slow", "empty", "decline", "declined", "cheaper", "drop"}


def economic_relevance_rerank(master_out, id_to_node, econ_ctx, weight=0.5):
  """Multiplicatively reweight scores by economic relevance. Identity when
  econ_ctx is falsy (economy off). Returns a NEW dict (does not mutate input).

  econ_ctx: {"cash": float|None, "daily_cost": float|None, "bankrupt": bool,
             "goal_resources": iterable[str]}  (any subset; missing -> neutral)
  """
  if not econ_ctx:
    return dict(master_out)
  goal_res = {r.lower() for r in (econ_ctx.get("goal_resources") or [])}
  cash = econ_ctx.get("cash")
  daily = econ_ctx.get("daily_cost") or 1.0
  urgent = bool(econ_ctx.get("bankrupt")) or (cash is not None and cash < 2 * daily)

  out = {}
  for nid, score in master_out.items():
    node = id_to_node.get(nid) if id_to_node else None
    boost = 0.0
    if node is not None:
      ann = typed_memory.annotate(node)
      if ann["mem_type"] == "economic":
        boost += weight * (2.0 if urgent else 1.0)
      overlap = goal_res & {r.lower() for r in (ann.get("resources") or [])}
      boost += 0.25 * len(overlap)
    out[nid] = score * (1.0 + boost)
  return out


def _polarity(node):
  text = (getattr(node, "description", "") or "").lower()
  toks = set(text.replace(".", " ").replace(",", " ").split())
  pos, neg = bool(toks & _POS_WORDS), bool(toks & _NEG_WORDS)
  if pos and not neg:
    return "pos"
  if neg and not pos:
    return "neg"
  return None  # mixed / neutral -> not used for contradiction


def contradiction_filter(master_out, id_to_node):
  """Drop candidates that contradict a higher-confidence memory on the SAME
  economic resource topic. Conservative: only fires for economic-typed nodes
  with a clear, opposite polarity; ties keep both. Returns (filtered_dict,
  dropped_ids). dropped_ids are candidates a caller may re-tag `invalidated`.

  No-op (drops nothing) when there are no contradicting economic pairs."""
  if not id_to_node:
    return dict(master_out), []

  # Bucket economic, polarized nodes by their resource topic.
  topics = {}  # frozenset(resources) -> list[(nid, polarity, confidence)]
  for nid in master_out:
    node = id_to_node.get(nid)
    if node is None:
      continue
    ann = typed_memory.annotate(node)
    if ann["mem_type"] != "economic" or not ann.get("resources"):
      continue
    pol = _polarity(node)
    if pol is None:
      continue
    topic = frozenset(r.lower() for r in ann["resources"])
    topics.setdefault(topic, []).append((nid, pol, ann["confidence"]))

  dropped = []
  for topic, items in topics.items():
    pols = {p for _, p, _ in items}
    if len(pols) < 2:
      continue  # no opposing views on this topic
    # The higher-confidence polarity is the "active belief"; drop opposite-
    # polarity candidates with STRICTLY lower confidence than it.
    best_conf = {p: max(c for _, pp, c in items if pp == p) for p in pols}
    belief_pol = max(best_conf, key=best_conf.get)
    belief_conf = best_conf[belief_pol]
    for nid, pol, conf in items:
      if pol != belief_pol and conf < belief_conf:
        dropped.append(nid)

  filtered = {nid: s for nid, s in master_out.items() if nid not in dropped}
  return filtered, dropped


def apply_rerank(master_out, id_to_node, econ_ctx):
  """Convenience: economic rerank then contradiction filter. Returns
  (reranked_filtered_dict, dropped_ids)."""
  reranked = economic_relevance_rerank(master_out, id_to_node, econ_ctx)
  return contradiction_filter(reranked, id_to_node)
