"""Dry-run harness for this repo's vedo-based scripts.

Runs another script's main() for a fixed number of timer ticks without
opening a real interactive window, optionally saving a screenshot at the
end. This is the exact monkeypatch pattern used throughout this project's
development to smoke-test with dummy/mock data before touching real
hardware -- it's what caught the vedo `.vertices = bigger_array` topology
bug (a resized Points/Lines actor silently only renders as many primitives
as it had at construction) in mocap_skeleton_vedo.py and
wrist_ik_test_vedo.py. Saved here as a real script instead of retyping the
monkeypatch each time.

How it works: vedo.Plotter.add_callback stashes the first "timer" callback
a script registers instead of really registering it, and
vedo.Plotter.show() calls that callback --ticks times (sleeping --sleep
between calls) instead of blocking for real interaction, then optionally
takes a screenshot.

Usage:
    python scripts/dev_dry_run.py mocap_skeleton_vedo -- --backend mock
    python scripts/dev_dry_run.py upper_body_retarget_vedo --ticks 200 --screenshot out.png -- --backend mock
    python scripts/dev_dry_run.py wrist_ik_test_vedo --ticks 300 --screenshot chart.png -- --hand left

Everything after a literal `--` is passed through as that script's own
argv (its own --backend/--hand/etc.); this tool's own --ticks/--sleep/
--screenshot must come before the `--`.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        own_argv, target_argv = argv[:split], argv[split + 1:]
    else:
        own_argv, target_argv = argv, []

    parser = argparse.ArgumentParser(
        description="Dry-run a scripts/*.py vedo tool's main() for N ticks with no real "
                     "window -- for smoke-testing with dummy data before real hardware.",
    )
    parser.add_argument("module", help="module name under scripts/, e.g. mocap_skeleton_vedo")
    parser.add_argument("--ticks", type=int, default=150, help="timer ticks to run")
    parser.add_argument("--sleep", type=float, default=0.005, help="seconds slept between ticks")
    parser.add_argument(
        "--screenshot", type=pathlib.Path, default=None,
        help="save a PNG here after the run (parent dirs created as needed)",
    )
    args = parser.parse_args(own_argv)

    # The target script's own argparse reads sys.argv itself -- point it at
    # just the pass-through args, same as if it had been run directly.
    sys.argv = [f"{args.module}.py", *target_argv]

    import vedo  # noqa: E402  (after sys.path setup above)

    captured = {"update": None}
    orig_add_callback = vedo.Plotter.add_callback

    def fake_add_callback(self, name, func, *a, **k):
        if name == "timer" and captured["update"] is None:
            # Every repo script here settled on exactly one "timer"
            # registration (vedo/VTK fires every registered "timer"
            # callback on every TimerEvent regardless of which dt created
            # it, so more than one would double-fire) -- capture that one.
            captured["update"] = func
            return 0
        return orig_add_callback(self, name, func, *a, **k)

    def fake_show(self, *a, **k):
        upd = captured["update"]
        if upd is None:
            print("[dev_dry_run] WARNING: no 'timer' callback was registered -- nothing to tick")
        else:
            for _ in range(args.ticks):
                upd(None)
                if args.sleep > 0:
                    time.sleep(args.sleep)
        if args.screenshot is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            self.screenshot(str(args.screenshot))
            print(f"[dev_dry_run] screenshot saved -> {args.screenshot}")
        return self

    vedo.Plotter.add_callback = fake_add_callback
    vedo.Plotter.show = fake_show

    mod = importlib.import_module(f"scripts.{args.module}")
    if not hasattr(mod, "main"):
        raise RuntimeError(f"scripts.{args.module} has no main() to run")

    t0 = time.perf_counter()
    try:
        mod.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    dt = time.perf_counter() - t0
    print(f"[dev_dry_run] {args.module}: {args.ticks} ticks OK in {dt:.2f}s, no exception raised")


if __name__ == "__main__":
    main()
