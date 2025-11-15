"""
Command-line interface for the Support Copilot Host.
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel

from .models import Ticket
from .orchestrator import run_ticket_flow
from .observability import render_trace_to_stdout, print_trace_summary, render_trace_as_jsonl
from .config import SAMPLE_TICKETS_PATH

app = typer.Typer(help="Support Copilot Host - AI-powered ticket handling")
console = Console()


@app.command()
def run_ticket(
    description: str = typer.Option(..., "--description", "-d", help="Ticket description"),
    screenshot: Optional[str] = typer.Option(None, "--screenshot", "-s", help="Path to screenshot"),
    log: Optional[str] = typer.Option(None, "--log", "-l", help="Path to log file or raw log text"),
    ticket_id: Optional[str] = typer.Option(None, "--id", help="Custom ticket ID")
):
    """
    Process a single support ticket.
    """
    # Generate ticket ID if not provided
    if not ticket_id:
        ticket_id = f"CLI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Handle log input (file or raw text)
    log_snippet = None
    if log:
        log_path = Path(log)
        if log_path.exists() and log_path.is_file():
            with open(log_path, "r") as f:
                log_snippet = f.read()
        else:
            log_snippet = log

    # Create ticket
    ticket = Ticket(
        id=ticket_id,
        description=description,
        screenshot_path=screenshot,
        log_snippet=log_snippet
    )

    # Run flow
    console.print(f"\n[bold cyan]Processing ticket {ticket_id}...[/bold cyan]\n")

    supervisor, trace = asyncio.run(run_ticket_flow(ticket))

    # Print results
    console.print(Panel(
        supervisor.final_customer_reply,
        title="[bold green]Customer Reply[/bold green]",
        border_style="green"
    ))

    console.print("\n")

    console.print(Panel(
        supervisor.final_internal_note,
        title="[bold yellow]Internal Note[/bold yellow]",
        border_style="yellow"
    ))

    console.print("\n")

    if not supervisor.approved:
        console.print(Panel(
            supervisor.review_notes,
            title="[bold red]Supervisor Review Notes[/bold red]",
            border_style="red"
        ))
        console.print("\n")

    # Print trace summary
    print_trace_summary(trace)

    # Save trace
    render_trace_as_jsonl(trace)
    console.print(f"\n[dim]Trace saved to {render_trace_as_jsonl.__globals__['TRACE_LOG_PATH']}[/dim]")


@app.command()
def run_sample(
    sample_id: str = typer.Option(..., "--id", help="Sample ticket ID (e.g., TICKET-001)")
):
    """
    Process a sample ticket from data/tickets/samples.json.
    """
    # Load samples
    if not SAMPLE_TICKETS_PATH.exists():
        console.print(f"[red]Error: Sample tickets file not found at {SAMPLE_TICKETS_PATH}[/red]")
        raise typer.Exit(1)

    with open(SAMPLE_TICKETS_PATH, "r") as f:
        samples = json.load(f)

    # Find ticket
    ticket_data = None
    for sample in samples:
        if sample["id"] == sample_id:
            ticket_data = sample
            break

    if not ticket_data:
        console.print(f"[red]Error: Sample ticket '{sample_id}' not found[/red]")
        console.print(f"Available tickets: {', '.join(s['id'] for s in samples)}")
        raise typer.Exit(1)

    # Create ticket
    ticket = Ticket(**ticket_data)

    # Run flow
    console.print(f"\n[bold cyan]Processing sample ticket {sample_id}...[/bold cyan]\n")

    supervisor, trace = asyncio.run(run_ticket_flow(ticket))

    # Print results
    console.print(Panel(
        supervisor.final_customer_reply,
        title="[bold green]Customer Reply[/bold green]",
        border_style="green"
    ))

    console.print("\n")

    console.print(Panel(
        supervisor.final_internal_note,
        title="[bold yellow]Internal Note[/bold yellow]",
        border_style="yellow"
    ))

    console.print("\n")

    if not supervisor.approved:
        console.print(Panel(
            supervisor.review_notes,
            title="[bold red]Supervisor Review Notes[/bold red]",
            border_style="red"
        ))
        console.print("\n")

    # Print trace
    render_trace_to_stdout(trace)

    # Save trace
    render_trace_as_jsonl(trace)
    console.print(f"\n[dim]Trace saved to trace log[/dim]")


@app.command()
def list_samples():
    """
    List available sample tickets.
    """
    if not SAMPLE_TICKETS_PATH.exists():
        console.print(f"[red]Error: Sample tickets file not found at {SAMPLE_TICKETS_PATH}[/red]")
        raise typer.Exit(1)

    with open(SAMPLE_TICKETS_PATH, "r") as f:
        samples = json.load(f)

    console.print("\n[bold cyan]Available Sample Tickets:[/bold cyan]\n")

    for sample in samples:
        console.print(f"[green]{sample['id']}[/green]")
        console.print(f"  {sample['description'][:80]}...")
        console.print()


if __name__ == "__main__":
    app()
