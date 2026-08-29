#!/usr/bin/env python3
"""Preflight, run, resume, or report the RelGen overnight session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overnight_session import OvernightSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/relgen_overnight_8h_session.yaml",
    )
    parser.add_argument("--stage", choices=("preflight", "run", "report"), default="preflight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--no-wait-for-deadline", action="store_true")
    parser.add_argument("--force-new-session", action="store_true")
    parser.add_argument("--allow-heartbeat-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    session = OvernightSession(
        Path(args.config),
        root=root,
        device=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        duration_hours=args.duration_hours,
        wait_for_deadline=not args.no_wait_for_deadline,
        force_new_session=args.force_new_session,
        allow_heartbeat_only=args.allow_heartbeat_only,
        preflight_only=args.stage == "preflight",
    )
    if args.stage == "preflight":
        report = session.preflight()
        print(session.display_path(session.output_dir / "preflight.json"))
        print(session.display_path(session.output_dir / "preflight.md"))
        if not report["launch_recommended"]:
            raise SystemExit(2)
    elif args.stage == "run":
        session.run()
    else:
        session.write_report(final=False)
        print(session.display_path(session.report_path))


if __name__ == "__main__":
    main()
