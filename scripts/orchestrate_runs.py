#!/usr/bin/env python3
"""Benchmark orchestrator for CausalArmor evaluation runs.

Manages 72 benchmark runs (3 providers x 4 suites x 2 modes x 3 repetitions)
with concurrency control, vLLM health monitoring, API rate-limit awareness,
and resume support.

Usage
-----
::

    # Dry run — show the full matrix and skip logic
    python scripts/orchestrate_runs.py --dry-run

    # Single-run smoke test
    python scripts/orchestrate_runs.py \\
        --filter-suite banking --filter-provider gemini --filter-mode baseline

    # Full launch with defaults
    python scripts/orchestrate_runs.py

"""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich is an optional but expected dependency in the test harness
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = ["gemini", "openai", "anthropic"]
SUITES = ["banking", "travel", "workspace", "slack"]
MODES = ["baseline", "guarded"]
REPETITIONS = 3

HARNESS_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = HARNESS_ROOT / "results"
LOGS_DIR = HARNESS_ROOT / "results" / "logs"
VENV_PYTHON = HARNESS_ROOT / ".venv" / "bin" / "python"

VLLM_HEALTH_URL = "http://localhost:8000/health"
VLLM_POLL_INTERVAL_S = 30
VLLM_FAILURE_THRESHOLD = 5

PROGRESS_INTERVAL_S = 60

# Per-provider concurrency defaults (reflects API RPM limits)
DEFAULT_PROVIDER_LIMITS: dict[str, int] = {
    "gemini": 2,  # 1000 RPM — 3 concurrent guarded runs caused 429s
    "openai": 2,  # 500 RPM
    "anthropic": 2,  # 50 RPM — can handle 2 concurrent
}

logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# RunSpec — immutable description of a single benchmark run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """Immutable specification for one benchmark run."""

    provider: str
    suite: str
    mode: str  # "baseline" or "guarded"
    run_number: int

    @property
    def needs_vllm(self) -> bool:
        return self.mode == "guarded"

    @property
    def output_filename(self) -> str:
        return f"{self.suite}_{self.provider}_{self.mode}_run{self.run_number}.json"

    @property
    def output_path(self) -> Path:
        return RESULTS_DIR / self.output_filename

    @property
    def log_path(self) -> Path:
        return LOGS_DIR / self.output_filename.replace(".json", ".log")

    @property
    def label(self) -> str:
        return f"{self.suite}/{self.provider}/{self.mode}/run{self.run_number}"


# ---------------------------------------------------------------------------
# RunState / RunStatus
# ---------------------------------------------------------------------------


class RunState(enum.Enum):
    PENDING = "pending"
    WAITING_VLLM = "waiting_vllm"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunStatus:
    """Mutable tracking state for a single run."""

    spec: RunSpec
    state: RunState = RunState.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    return_code: int | None = None
    scenario_count: int | None = None
    error_count: int | None = None
    failure_reason: str | None = None

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at else time.monotonic()
        return end - self.started_at

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e is None:
            return "-"
        m, s = divmod(int(e), 60)
        return f"{m:d}m{s:02d}s"


# ---------------------------------------------------------------------------
# ConcurrencyController
# ---------------------------------------------------------------------------


class ConcurrencyController:
    """Manages heterogeneous concurrency constraints.

    Uses ``asyncio.Condition`` + counters instead of nested semaphores to
    avoid potential deadlocks.
    """

    def __init__(
        self,
        max_total: int,
        max_baseline: int,
        max_guarded: int,
        provider_limits: dict[str, int],
    ) -> None:
        self._cond = asyncio.Condition()
        self._max_total = max_total
        self._max_baseline = max_baseline
        self._max_guarded = max_guarded
        self._provider_limits = dict(provider_limits)

        self._total = 0
        self._baseline = 0
        self._guarded = 0
        self._per_provider: dict[str, int] = {p: 0 for p in PROVIDERS}

    def _can_acquire(self, spec: RunSpec) -> bool:
        if self._total >= self._max_total:
            return False
        if spec.mode == "baseline" and self._baseline >= self._max_baseline:
            return False
        if spec.mode == "guarded" and self._guarded >= self._max_guarded:
            return False
        limit = self._provider_limits.get(spec.provider, self._max_total)
        if self._per_provider.get(spec.provider, 0) >= limit:
            return False
        return True

    async def acquire(self, spec: RunSpec) -> None:
        async with self._cond:
            while not self._can_acquire(spec):
                await self._cond.wait()
            self._total += 1
            if spec.mode == "baseline":
                self._baseline += 1
            else:
                self._guarded += 1
            self._per_provider[spec.provider] = (
                self._per_provider.get(spec.provider, 0) + 1
            )

    async def release(self, spec: RunSpec) -> None:
        async with self._cond:
            self._total -= 1
            if spec.mode == "baseline":
                self._baseline -= 1
            else:
                self._guarded -= 1
            self._per_provider[spec.provider] = (
                self._per_provider.get(spec.provider, 0) - 1
            )
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# vLLM Health Monitor
# ---------------------------------------------------------------------------


class VLLMHealthMonitor:
    """Background task that polls the vLLM health endpoint.

    Guarded runs ``await wait_healthy()`` before launching.
    """

    def __init__(self) -> None:
        self._healthy = asyncio.Event()
        self._healthy.set()  # optimistic start
        self._consecutive_failures = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def wait_healthy(self) -> None:
        await self._healthy.wait()

    @property
    def is_healthy(self) -> bool:
        return self._healthy.is_set()

    async def _poll_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ok = await loop.run_in_executor(None, self._check_health)
                if ok:
                    if self._consecutive_failures > 0:
                        logger.info(
                            "vLLM recovered after %d consecutive failures",
                            self._consecutive_failures,
                        )
                    self._consecutive_failures = 0
                    self._healthy.set()
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= VLLM_FAILURE_THRESHOLD:
                        if self._healthy.is_set():
                            logger.warning(
                                "vLLM unhealthy (%d consecutive failures) "
                                "— pausing guarded runs",
                                self._consecutive_failures,
                            )
                        self._healthy.clear()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= VLLM_FAILURE_THRESHOLD:
                    self._healthy.clear()

            await asyncio.sleep(VLLM_POLL_INTERVAL_S)

    @staticmethod
    def _check_health() -> bool:
        try:
            req = urllib.request.Request(VLLM_HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Run matrix generation
# ---------------------------------------------------------------------------


def build_run_matrix(
    filter_provider: str | None = None,
    filter_suite: str | None = None,
    filter_mode: str | None = None,
) -> list[RunSpec]:
    """Build the full 72-run matrix with optional filters."""
    specs: list[RunSpec] = []
    providers = [filter_provider] if filter_provider else PROVIDERS
    suites = [filter_suite] if filter_suite else SUITES
    modes = [filter_mode] if filter_mode else MODES

    for run_number in range(1, REPETITIONS + 1):
        for suite in suites:
            for provider in providers:
                for mode in modes:
                    specs.append(RunSpec(provider, suite, mode, run_number))
    return specs


def order_runs(specs: list[RunSpec]) -> list[RunSpec]:
    """Order runs: baselines first, then guarded; round-robin providers."""
    baselines = [s for s in specs if s.mode == "baseline"]
    guarded = [s for s in specs if s.mode == "guarded"]

    def _round_robin(items: list[RunSpec]) -> list[RunSpec]:
        """Interleave by provider to spread API load."""
        by_provider: dict[str, list[RunSpec]] = {}
        for item in items:
            by_provider.setdefault(item.provider, []).append(item)
        result: list[RunSpec] = []
        queues = [by_provider[p] for p in PROVIDERS if p in by_provider]
        idx = 0
        while queues:
            q = queues[idx % len(queues)]
            result.append(q.pop(0))
            if not q:
                queues.remove(q)
            else:
                idx += 1
        return result

    return [*_round_robin(baselines), *_round_robin(guarded)]


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _build_command(spec: RunSpec, *, use_openrouter: bool = False) -> list[str]:
    """Build the subprocess command for a benchmark run."""
    guard_flag = "--guard" if spec.mode == "guarded" else "--no-guard"
    cmd = [
        str(VENV_PYTHON),
        "-m",
        "agentdojo_native.run_benchmark",
        "--provider",
        spec.provider,
        "--suite",
        spec.suite,
        "--attack",
        "important_instructions",
        guard_flag,
        "-o",
        str(spec.output_path),
    ]
    if use_openrouter:
        cmd.append("--openrouter")
    return cmd


def _parse_result_file(path: Path) -> tuple[int, int]:
    """Parse a result JSON file for scenario_count and error_count."""
    try:
        with open(path) as f:
            data = json.load(f)
        # Result is a list with one suite entry
        entry = data[0] if isinstance(data, list) else data
        scenarios = entry.get("scenarios", [])
        n = len(scenarios)
        errors = sum(1 for s in scenarios if s.get("error"))
        return n, errors
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Manages concurrent benchmark subprocess execution."""

    def __init__(
        self,
        specs: list[RunSpec],
        concurrency: ConcurrencyController,
        vllm_monitor: VLLMHealthMonitor,
        dry_run: bool = False,
        log_file: Path | None = None,
        use_openrouter: bool = False,
    ) -> None:
        self._specs = specs
        self._concurrency = concurrency
        self._vllm = vllm_monitor
        self._dry_run = dry_run
        self._log_file = log_file
        self._use_openrouter = use_openrouter

        self._statuses: list[RunStatus] = [RunStatus(spec=s) for s in specs]
        self._tasks: list[asyncio.Task[None]] = []
        self._draining = False  # True after first signal — no new launches
        self._force_shutdown = False  # True after second signal — kill everything

    # -- public API --

    async def run(self) -> list[RunStatus]:
        """Execute all runs and return final statuses."""
        if self._dry_run:
            self._show_dry_run()
            return self._statuses

        # Start vLLM monitor if any guarded runs
        has_guarded = any(s.spec.needs_vllm for s in self._statuses)
        if has_guarded:
            await self._vllm.start()

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._handle_shutdown())

        # Start progress reporter
        progress_task = asyncio.create_task(self._progress_loop())

        # Launch all runs as concurrent tasks
        for status in self._statuses:
            task = asyncio.create_task(self._execute_run(status))
            self._tasks.append(task)

        # Wait for all to finish
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Cleanup
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        if has_guarded:
            await self._vllm.stop()

        self._show_summary()
        return self._statuses

    # -- run execution --

    async def _execute_run(self, status: RunStatus) -> None:
        spec = status.spec

        # Resume support: skip if output file already exists
        if spec.output_path.exists():
            status.state = RunState.SKIPPED
            sc, ec = _parse_result_file(spec.output_path)
            status.scenario_count = sc
            status.error_count = ec
            logger.info("SKIP  %s  (output exists, %d scenarios)", spec.label, sc)
            return

        slot_acquired = False
        try:
            # Drain check: don't start new work after shutdown requested
            if self._draining:
                status.state = RunState.SKIPPED
                status.failure_reason = "skipped (drain)"
                logger.info("DRAIN %s  (shutdown requested, not starting)", spec.label)
                return

            # Wait for vLLM if guarded
            if spec.needs_vllm:
                status.state = RunState.WAITING_VLLM
                logger.info("WAIT  %s  (waiting for vLLM)", spec.label)
                await self._vllm.wait_healthy()

            # Check again after potentially long wait
            if self._draining:
                status.state = RunState.SKIPPED
                status.failure_reason = "skipped (drain)"
                logger.info("DRAIN %s  (shutdown requested, not starting)", spec.label)
                return

            # Acquire concurrency slot
            status.state = RunState.PENDING
            await self._concurrency.acquire(spec)
            slot_acquired = True

            # Check once more after acquiring slot
            if self._draining:
                status.state = RunState.SKIPPED
                status.failure_reason = "skipped (drain)"
                logger.info("DRAIN %s  (shutdown requested, not starting)", spec.label)
                return

            # Launch subprocess with per-run log file
            status.state = RunState.RUNNING
            status.started_at = time.monotonic()
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_path = spec.log_path
            logger.info("START %s  (log: %s)", spec.label, log_path)

            cmd = _build_command(
                spec, use_openrouter=self._use_openrouter,
            )
            log_fh = open(log_path, "w")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(HARNESS_ROOT / "src"),
            )
            status.process = proc

            await proc.wait()
            log_fh.close()
            status.return_code = proc.returncode
            status.finished_at = time.monotonic()

            if proc.returncode == 0:
                status.state = RunState.COMPLETED
                sc, ec = _parse_result_file(spec.output_path)
                status.scenario_count = sc
                status.error_count = ec
                logger.info(
                    "DONE  %s  (%s, %d scenarios, %d errors)",
                    spec.label,
                    status.elapsed_str,
                    sc,
                    ec,
                )
            else:
                status.state = RunState.FAILED
                # Capture last 20 lines of log for diagnostics
                output_tail = ""
                try:
                    lines = log_path.read_text().strip().splitlines()
                    output_tail = "\n".join(lines[-20:])
                except Exception:
                    pass
                status.failure_reason = (
                    f"exit code {proc.returncode}\n{output_tail}"
                )
                logger.error(
                    "FAIL  %s  (exit code %d, %s)",
                    spec.label,
                    proc.returncode,
                    status.elapsed_str,
                )
                if output_tail:
                    logger.error("  Last output:\n%s", output_tail)

        except asyncio.CancelledError:
            status.finished_at = time.monotonic()
            if status.process and status.process.returncode is None:
                # Was running a subprocess — force-killed
                status.process.terminate()
                status.state = RunState.FAILED
                status.failure_reason = "terminated (force shutdown)"
            else:
                # Was waiting (vLLM / concurrency slot) — clean skip
                status.state = RunState.SKIPPED
                status.failure_reason = "skipped (drain)"
            raise
        except Exception as exc:
            status.state = RunState.FAILED
            status.failure_reason = str(exc)
            status.finished_at = time.monotonic()
            logger.exception("ERROR %s: %s", spec.label, exc)
        finally:
            if slot_acquired:
                await self._concurrency.release(spec)

    # -- shutdown --

    def _handle_shutdown(self) -> None:
        if self._force_shutdown:
            # Third signal — nothing more we can do
            return

        if not self._draining:
            # First signal: drain — stop launching new runs, let active ones finish
            self._draining = True
            running_count = sum(
                1 for s in self._statuses if s.state == RunState.RUNNING
            )
            logger.warning(
                "Shutdown requested — draining. "
                "%d active run(s) will finish. "
                "Signal again to force-kill.",
                running_count,
            )
            # Cancel tasks that haven't started a subprocess yet
            for task, status in zip(self._tasks, self._statuses, strict=True):
                if not task.done() and status.state in (
                    RunState.PENDING,
                    RunState.WAITING_VLLM,
                ):
                    task.cancel()
        else:
            # Second signal: force — terminate running subprocesses
            self._force_shutdown = True
            logger.warning("Force shutdown — terminating all subprocesses...")
            for task, status in zip(self._tasks, self._statuses, strict=True):
                if not task.done():
                    if status.process and status.process.returncode is None:
                        status.process.terminate()
                    task.cancel()

    # -- progress reporting --

    async def _progress_loop(self) -> None:
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL_S)
            self._log_progress()

    def _log_progress(self) -> None:
        counts: dict[str, int] = {}
        for s in self._statuses:
            counts[s.state.value] = counts.get(s.state.value, 0) + 1

        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        logger.info("Progress: %s", "  ".join(parts))

        active = [s for s in self._statuses if s.state == RunState.RUNNING]
        for s in active:
            logger.info("  ACTIVE  %-40s  %s", s.spec.label, s.elapsed_str)

    # -- display --

    def _show_dry_run(self) -> None:
        skipped = []
        pending = []
        for status in self._statuses:
            if status.spec.output_path.exists():
                status.state = RunState.SKIPPED
                skipped.append(status)
            else:
                pending.append(status)

        print(f"\n{'='*70}")
        print(f"  DRY RUN — {len(self._statuses)} total runs")
        print(f"  {len(pending)} to execute, {len(skipped)} to skip (output exists)")
        print(f"{'='*70}\n")

        if pending:
            print("Runs to execute:")
            for i, s in enumerate(pending, 1):
                vllm_tag = " [vLLM]" if s.spec.needs_vllm else ""
                print(f"  {i:3d}. {s.spec.label}{vllm_tag}")
                print(f"       → {s.spec.output_path}")

        if skipped:
            print(f"\nRuns to skip ({len(skipped)}):")
            for s in skipped:
                sc, ec = _parse_result_file(s.spec.output_path)
                print(f"  SKIP {s.spec.label}  ({sc} scenarios)")

        print()

    def _show_summary(self) -> None:
        completed = [s for s in self._statuses if s.state == RunState.COMPLETED]
        failed = [s for s in self._statuses if s.state == RunState.FAILED]
        skipped = [s for s in self._statuses if s.state == RunState.SKIPPED]

        print(f"\n{'='*70}")
        print(f"  SUMMARY: {len(completed)} completed, {len(failed)} failed, "
              f"{len(skipped)} skipped")
        print(f"{'='*70}\n")

        if HAS_RICH:
            self._show_rich_table()
        else:
            self._show_plain_table()

        if failed:
            print("\nFailed runs:")
            for s in failed:
                print(f"  {s.spec.label}: {s.failure_reason}")

    def _show_rich_table(self) -> None:
        console = Console()
        table = Table(title="Benchmark Results")
        table.add_column("Run", style="cyan")
        table.add_column("State", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Scenarios", justify="right")
        table.add_column("Errors", justify="right")

        state_styles = {
            RunState.COMPLETED: "green",
            RunState.FAILED: "red bold",
            RunState.SKIPPED: "yellow",
            RunState.PENDING: "dim",
            RunState.RUNNING: "blue",
            RunState.WAITING_VLLM: "magenta",
        }

        for s in self._statuses:
            style = state_styles.get(s.state, "")
            table.add_row(
                s.spec.label,
                f"[{style}]{s.state.value}[/{style}]",
                s.elapsed_str,
                str(s.scenario_count or "-"),
                str(s.error_count or "-"),
            )

        console.print(table)

    def _show_plain_table(self) -> None:
        header = f"{'Run':<45s} {'State':<12s} {'Duration':>10s} {'Scen':>6s} {'Err':>5s}"
        print(header)
        print("-" * len(header))
        for s in self._statuses:
            print(
                f"{s.spec.label:<45s} {s.state.value:<12s} "
                f"{s.elapsed_str:>10s} "
                f"{str(s.scenario_count or '-'):>6s} "
                f"{str(s.error_count or '-'):>5s}"
            )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path | None, verbose: bool = False) -> None:
    """Configure dual logging: file (full detail) + console (summary)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # File handler — full detail
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info("Logging to %s", log_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate CausalArmor benchmark runs with concurrency control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-baseline",
        type=int,
        default=4,
        help="Max concurrent baseline runs.",
    )
    parser.add_argument(
        "--max-guarded",
        type=int,
        default=5,
        help="Max concurrent guarded runs.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=8,
        help="Max total concurrent runs.",
    )
    parser.add_argument(
        "--filter-provider",
        choices=PROVIDERS,
        default=None,
        help="Only run this provider.",
    )
    parser.add_argument(
        "--filter-suite",
        choices=SUITES,
        default=None,
        help="Only run this suite.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=MODES,
        default=None,
        help="Only run baseline or guarded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(__file__).resolve().parent / "orchestrator.log",
        help="Log file path.",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Route provider calls through OpenRouter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging to console.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> None:
    args = parse_args()
    setup_logging(args.log_file, verbose=args.verbose)

    logger.info("CausalArmor Benchmark Orchestrator")
    logger.info(
        "Concurrency limits: total=%d  baseline=%d  guarded=%d",
        args.max_total,
        args.max_baseline,
        args.max_guarded,
    )
    logger.info(
        "Provider limits: %s",
        "  ".join(f"{p}={DEFAULT_PROVIDER_LIMITS[p]}" for p in PROVIDERS),
    )

    # Build and order the run matrix
    specs = build_run_matrix(
        filter_provider=args.filter_provider,
        filter_suite=args.filter_suite,
        filter_mode=args.filter_mode,
    )
    ordered = order_runs(specs)

    logger.info("Total runs in matrix: %d", len(ordered))

    # Setup components
    concurrency = ConcurrencyController(
        max_total=args.max_total,
        max_baseline=args.max_baseline,
        max_guarded=args.max_guarded,
        provider_limits=DEFAULT_PROVIDER_LIMITS,
    )
    vllm_monitor = VLLMHealthMonitor()

    orchestrator = Orchestrator(
        specs=ordered,
        concurrency=concurrency,
        vllm_monitor=vllm_monitor,
        dry_run=args.dry_run,
        log_file=args.log_file,
        use_openrouter=args.openrouter,
    )

    results = await orchestrator.run()

    # Exit code: non-zero if any runs failed
    failed = sum(1 for r in results if r.state == RunState.FAILED)
    if failed:
        logger.error("%d run(s) FAILED", failed)
        sys.exit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
