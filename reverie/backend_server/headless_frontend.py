"""Headless frontend stub.

Replaces the Django visualizer for unattended runs. Polls
`storage/<sim>/movement/<n>.json` and immediately writes the corresponding
`storage/<sim>/environment/<n+1>.json` so the backend can advance its main
loop without a browser.

We do not simulate spatial collisions or pathing -- agents just teleport to
the tile the backend already chose for them. This matches what the Django
frontend does when it loads movement and advances the world clock; the
spatial reasoning that uses path_finder happens inside the backend's
execute module before movement is written, so the stub does not need to
reproduce it.

Usage:
  python -m headless_frontend --sim <sim_code> [--poll 0.5] [--max-step 0]
  (max-step=0 means run until reverie writes a "fin" sentinel or the
   process is killed by the supervisor.)
"""
import argparse
import json
import os
import sys
import time

DEFAULT_POLL = 0.5
STORAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "environment", "frontend_server", "storage",
)


def storage_dir(sim_code):
  return os.path.abspath(os.path.join(STORAGE_ROOT, sim_code))


def movement_path(sim, step):
  return os.path.join(storage_dir(sim), "movement", f"{step}.json")


def env_path(sim, step):
  return os.path.join(storage_dir(sim), "environment", f"{step}.json")


def write_next_env(sim, movement_step, maze_name):
  """Convert movement/<step>.json to environment/<step+1>.json."""
  with open(movement_path(sim, movement_step)) as f:
    mv = json.load(f)
  persona = mv.get("persona", {})
  out = {}
  for name, info in persona.items():
    tile = info.get("movement") or [0, 0]
    out[name] = {"maze": maze_name, "x": tile[0], "y": tile[1]}
  dst = env_path(sim, movement_step + 1)
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  with open(dst, "w") as f:
    f.write(json.dumps(out, indent=2))
  return dst


def discover_maze(sim):
  """Read maze_name from meta.json; fall back to existing env/0.json."""
  try:
    with open(os.path.join(storage_dir(sim), "reverie", "meta.json")) as f:
      return json.load(f)["maze_name"]
  except Exception:
    pass
  try:
    with open(env_path(sim, 0)) as f:
      env0 = json.load(f)
    for name, info in env0.items():
      m = info.get("maze")
      if m:
        return m
  except Exception:
    pass
  return "the_ville"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sim", required=True)
  ap.add_argument("--poll", type=float, default=DEFAULT_POLL)
  ap.add_argument("--max-step", type=int, default=0,
                  help="Stop after this many movement files processed "
                       "(0 = run indefinitely until killed)")
  ap.add_argument("--idle-exit-seconds", type=float, default=1800.0,
                  help="If no new movement file for this long, exit. "
                       "Default 30 min.")
  args = ap.parse_args()

  sim = args.sim
  maze = discover_maze(sim)
  print(f"[stub] sim={sim} maze={maze} poll={args.poll}s "
        f"idle_exit={args.idle_exit_seconds}s", flush=True)

  # Skip steps that already have env file (resume-friendly).
  next_step_to_process = 0
  while os.path.exists(env_path(sim, next_step_to_process + 1)):
    next_step_to_process += 1
  if next_step_to_process > 0:
    print(f"[stub] resume: next movement file to process = "
          f"{next_step_to_process}", flush=True)

  processed = 0
  last_activity_ts = time.time()
  while True:
    mp = movement_path(sim, next_step_to_process)
    if os.path.exists(mp):
      try:
        dst = write_next_env(sim, next_step_to_process, maze)
      except Exception as e:
        # Movement file may be mid-write; try again next tick.
        print(f"[stub] step={next_step_to_process} write failed: {e!r}",
              flush=True)
        time.sleep(args.poll)
        continue
      processed += 1
      last_activity_ts = time.time()
      if processed % 100 == 0 or processed < 5:
        print(f"[stub] wrote {os.path.basename(dst)} "
              f"(processed={processed})", flush=True)
      next_step_to_process += 1
      if args.max_step and processed >= args.max_step:
        print(f"[stub] reached max-step ({args.max_step}), exiting",
              flush=True)
        return
      continue

    # No new movement yet -- check idle timeout.
    if (args.idle_exit_seconds > 0 and
        time.time() - last_activity_ts > args.idle_exit_seconds):
      print(f"[stub] no new movement for "
            f"{args.idle_exit_seconds:.0f}s, exiting", flush=True)
      return
    time.sleep(args.poll)


if __name__ == "__main__":
  main()
