"""
Example functions for workshop demonstrations.
"""
import asyncio
import json
from rich.console import Console
from rich.panel import Panel

from .models import Ticket
from .orchestrator import run_ticket_flow
from .observability import print_trace_summary, render_trace_to_stdout
from .config import SAMPLE_TICKETS_PATH

console = Console()


def _load_sample_ticket(ticket_id: str) -> Ticket:
    """Load a sample ticket by ID."""
    with open(SAMPLE_TICKETS_PATH, "r") as f:
        samples = json.load(f)

    for sample in samples:
        if sample["id"] == ticket_id:
            return Ticket(**sample)

    raise ValueError(f"Sample ticket {ticket_id} not found")


async def _run_and_display(ticket: Ticket):
    """Run ticket flow and display results."""
    console.print(f"\n[bold cyan]Processing ticket {ticket.id}...[/bold cyan]")
    console.print(f"[dim]{ticket.description}[/dim]\n")

    supervisor, trace = await run_ticket_flow(ticket)

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

    # Print trace summary
    print_trace_summary(trace)

    console.print("\n" + "="*80 + "\n")


def demo_known_issue():
    """
    Demonstrate handling of a known issue (export timeout).

    Uses TICKET-001: Export to CSV timeout
    """
    console.print("\n[bold magenta]Demo: Known Issue (Export Timeout)[/bold magenta]\n")

    ticket = _load_sample_ticket("TICKET-001")
    asyncio.run(_run_and_display(ticket))


def demo_outage_case():
    """
    Demonstrate handling of a potential outage case.

    Uses TICKET-002: SSO authentication failures
    """
    console.print("\n[bold magenta]Demo: Outage Case (SSO Failures)[/bold magenta]\n")

    ticket = _load_sample_ticket("TICKET-002")
    asyncio.run(_run_and_display(ticket))


def demo_config_error():
    """
    Demonstrate handling of a configuration error.

    Uses TICKET-003: Workspace permissions issue
    """
    console.print("\n[bold magenta]Demo: Config Error (Permissions)[/bold magenta]\n")

    ticket = _load_sample_ticket("TICKET-003")
    asyncio.run(_run_and_display(ticket))


def demo_all():
    """Run all demo scenarios."""
    console.print("\n[bold white on blue] Running All Demo Scenarios [/bold white on blue]\n")

    demo_known_issue()
    demo_outage_case()
    demo_config_error()

    console.print("[bold green]All demos completed![/bold green]\n")


if __name__ == "__main__":
    # Run all demos when executed directly
    demo_all()
