#!/usr/bin/env python3
"""Live plaquette plotter for mcmc.py stdout/log streams.

Parses lines like:
  meas k=123 plaquette=0.6123 ...
and updates a live plot of plaquette vs measurement step.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple


MEAS_RE = re.compile(
    r"meas\s+k=(?P<k>\d+).*?\bplaquette=(?P<p>(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|nan|inf|-inf)"
)


def _iter_lines_file(path: Path, follow: bool, poll_sec: float) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                yield line
                continue
            if not follow:
                break
            time.sleep(max(0.01, float(poll_sec)))


def _iter_lines_stdin() -> Iterator[str]:
    for line in sys.stdin:
        yield line


def _parse_meas(line: str) -> Optional[Tuple[int, float]]:
    m = MEAS_RE.search(line)
    if m is None:
        return None
    try:
        k = int(m.group("k"))
        p = float(m.group("p"))
    except Exception:
        return None
    return k, p


def _compute_ylim(yvals):
    ys = [float(y) for y in yvals if math.isfinite(float(y))]
    if not ys:
        return (-1.0, 1.0)
    y0 = min(ys)
    y1 = max(ys)
    if y0 == y1:
        dy = max(1e-6, abs(y0) * 1e-3)
        return (y0 - dy, y1 + dy)
    pad = 0.05 * (y1 - y0)
    return (y0 - pad, y1 + pad)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live plot plaquette from mcmc.py output")
    ap.add_argument("--input", type=str, default="-", help="input log path, or '-' for stdin")
    ap.add_argument("--follow", action=argparse.BooleanOptionalAction, default=False, help="follow input file as it grows")
    ap.add_argument("--poll-sec", type=float, default=0.2, help="poll interval when --follow is used")
    ap.add_argument("--refresh-every", type=int, default=1, help="plot refresh cadence in accepted plaquette points")
    ap.add_argument("--echo", action=argparse.BooleanOptionalAction, default=False, help="echo incoming lines to stdout")
    ap.add_argument("--title", type=str, default="Plaquette Thermalization")
    ap.add_argument("--save", type=str, default="", help="optional output PNG path (updated periodically)")
    ap.add_argument("--csv", type=str, default="", help="optional CSV output path for parsed points")
    ap.add_argument("--no-gui", action="store_true", help="disable GUI; useful for headless runs")
    args = ap.parse_args()

    if "MPLCONFIGDIR" not in os.environ:
        mpldir = Path(tempfile.gettempdir()) / f"mplconfig_{os.getuid()}"
        mpldir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpldir)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg = Path(tempfile.gettempdir()) / f"xdg_cache_{os.getuid()}"
        xdg.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg)

    if args.no_gui:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if args.refresh_every <= 0:
        raise ValueError("--refresh-every must be >= 1")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    line, = ax.plot([], [], lw=1.5)
    ax.set_xlabel("measurement step k")
    ax.set_ylabel("plaquette")
    ax.set_title(str(args.title))
    ax.grid(True, alpha=0.25)

    if not args.no_gui:
        plt.ion()
        plt.show(block=False)

    csv_f = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_f = csv_path.open("w", encoding="utf-8")
        csv_f.write("k,plaquette\n")
        csv_f.flush()

    save_path = Path(args.save) if args.save else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    ks = []
    ps = []

    if args.input == "-":
        lines = _iter_lines_stdin()
    else:
        lines = _iter_lines_file(Path(args.input), follow=bool(args.follow), poll_sec=float(args.poll_sec))

    def refresh_plot(force_save: bool = False):
        if not ks:
            return
        line.set_data(ks, ps)
        x0 = min(ks)
        x1 = max(ks)
        if x0 == x1:
            x0 -= 1
            x1 += 1
        ax.set_xlim(x0, x1)
        y0, y1 = _compute_ylim(ps)
        ax.set_ylim(y0, y1)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)
        if not args.no_gui:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        if save_path is not None and (force_save or (len(ks) % max(5, int(args.refresh_every)) == 0)):
            fig.savefig(save_path, dpi=140, bbox_inches="tight")

    n_parsed = 0
    try:
        for line_in in lines:
            if args.echo:
                sys.stdout.write(line_in)
                sys.stdout.flush()
            parsed = _parse_meas(line_in)
            if parsed is None:
                continue
            k, p = parsed
            ks.append(int(k))
            ps.append(float(p))
            n_parsed += 1
            if csv_f is not None:
                csv_f.write(f"{int(k)},{float(p):.17g}\n")
                csv_f.flush()
            if n_parsed % int(args.refresh_every) == 0:
                refresh_plot(force_save=False)
    except KeyboardInterrupt:
        pass
    finally:
        refresh_plot(force_save=True)
        if csv_f is not None:
            csv_f.close()

    if save_path is not None and not ks:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")

    sys.stderr.write(f"[live_plot_plaquette] parsed points: {n_parsed}\n")
    sys.stderr.flush()

    if not args.no_gui:
        try:
            plt.ioff()
            plt.show()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
