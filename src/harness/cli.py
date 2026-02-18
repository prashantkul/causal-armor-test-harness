"""Typer CLI entry point for the CausalArmor benchmark harness."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="ca-bench",
    help="CausalArmor AgentDojo benchmark harness.",
    no_args_is_help=True,
)
console = Console()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def run(
    suite: Optional[str] = typer.Option(
        None, "--suite", "-s", help="Single suite to run (workspace/travel/banking/slack)."
    ),
    user_task: Optional[str] = typer.Option(
        None, "--user-task", "-u", help="Filter to a single user task ID."
    ),
    injection_task: Optional[str] = typer.Option(
        None, "--injection-task", "-i", help="Filter to a single injection task ID."
    ),
    no_guard: bool = typer.Option(
        False, "--no-guard", help="Run without CausalArmor (baseline)."
    ),
    attack: Optional[str] = typer.Option(
        None, "--attack", "-a", help="Attack name to use (e.g. important_instructions)."
    ),
    agent_model: str = typer.Option(
        "gemini-2.0-flash", "--agent-model", help="Agent LLM model."
    ),
    benchmark_version: str = typer.Option(
        "v1.1", "--version", "-v", help="AgentDojo benchmark version."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="JSON output path."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Run benchmark scenarios."""
    _setup_logging(verbose)

    from harness.report import report
    from harness.runner import run_full_benchmark, run_suite

    guard_enabled = not no_guard

    # Load attack if specified
    attack_obj = None
    if attack:
        from agentdojo.attacks import load_attack
        from agentdojo.task_suite.load_suites import get_suite

        # Attack needs a task suite and a target pipeline; we pass a
        # placeholder pipeline since we don't use AgentDojo's pipeline.
        target_suite = get_suite(benchmark_version, suite or "banking")
        attack_obj = load_attack(attack, target_suite, target_pipeline=None)

    # Determine output path
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        guard_label = "guarded" if guard_enabled else "baseline"
        output = RESULTS_DIR / f"benchmark_{guard_label}_{timestamp}.json"

    console.print(
        f"\n[bold]CausalArmor Benchmark[/bold]  "
        f"guard={'ON' if guard_enabled else 'OFF'}  "
        f"agent={agent_model}"
    )

    if suite and (user_task or injection_task):
        # Single suite with optional filters
        suite_result = asyncio.run(
            run_suite(
                suite,
                benchmark_version=benchmark_version,
                attack=attack_obj,
                guard_enabled=guard_enabled,
                agent_model=agent_model,
                user_task_filter=user_task,
                injection_task_filter=injection_task,
            )
        )
        report([suite_result], output_path=output)
    elif suite:
        suite_result = asyncio.run(
            run_suite(
                suite,
                benchmark_version=benchmark_version,
                attack=attack_obj,
                guard_enabled=guard_enabled,
                agent_model=agent_model,
            )
        )
        report([suite_result], output_path=output)
    else:
        results = asyncio.run(
            run_full_benchmark(
                benchmark_version=benchmark_version,
                attack=attack_obj,
                guard_enabled=guard_enabled,
                agent_model=agent_model,
                suite_filter=suite,
            )
        )
        report(results, output_path=output)


@app.command()
def classify(
    suite_name: str = typer.Argument(help="Suite to classify (workspace/travel/banking/slack)."),
    benchmark_version: str = typer.Option(
        "v1.1", "--version", "-v", help="AgentDojo benchmark version."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """Show untrusted tool classification for a suite."""
    _setup_logging(verbose)

    from agentdojo.task_suite.load_suites import get_suite

    from harness.classify import classify_untrusted_tools

    suite = get_suite(benchmark_version, suite_name)
    untrusted = classify_untrusted_tools(suite)

    table = Table(title=f"Tool Classification: {suite_name}")
    table.add_column("Tool", style="bold")
    table.add_column("Classification")

    for tool in sorted(suite.tools, key=lambda t: t.name):
        if tool.name in untrusted:
            table.add_row(tool.name, "[red]UNTRUSTED[/red]")
        else:
            table.add_row(tool.name, "[green]TRUSTED[/green]")

    console.print(table)


@app.command(name="list-tasks")
def list_tasks(
    suite_name: str = typer.Argument(help="Suite to list tasks for."),
    benchmark_version: str = typer.Option(
        "v1.1", "--version", "-v", help="AgentDojo benchmark version."
    ),
) -> None:
    """List available user tasks and injection tasks in a suite."""
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(benchmark_version, suite_name)

    console.print(f"\n[bold cyan]User Tasks ({suite_name})[/bold cyan]")
    for tid, task in sorted(suite.user_tasks.items()):
        prompt_preview = task.PROMPT[:80] + "..." if len(task.PROMPT) > 80 else task.PROMPT
        console.print(f"  {tid}: {prompt_preview}")

    console.print(f"\n[bold cyan]Injection Tasks ({suite_name})[/bold cyan]")
    for tid, task in sorted(suite.injection_tasks.items()):
        console.print(f"  {tid}: {task.GOAL}")

    console.print(
        f"\nTotal: {len(suite.user_tasks)} user tasks x "
        f"{len(suite.injection_tasks)} injection tasks = "
        f"{len(suite.user_tasks) * len(suite.injection_tasks)} scenarios"
    )


if __name__ == "__main__":
    app()
