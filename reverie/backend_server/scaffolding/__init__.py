"""C2.5 typed scaffolding package (Structured Scaffolding without Strategy).

Houses the general structured scaffolding shared by conditions C2.5 / C3 / C4
(and the typed action API, also shared by C1): constraint planner / validator,
typed action API, typed-memory helpers and retrieval rerank steps.

Design: 04-testbed-evolutionville/design/C2.5-typed-scaffolding.md
All modules here are deterministic (no LLM/RNG) unless explicitly noted, so the
constraint planner and action translation stay seed-reproducible. Like the
economy package, scaffolding never raises into the simulation loop: callers get
an explicit result object and decide what to do.
"""
