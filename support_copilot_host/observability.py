"""
Observability utilities for tracing and logging.
"""
import json
from typing import Optional
from rich.console import Console
from rich.table import Table

from .models import Trace
from .config import TRACE_LOG_PATH

console = Console()


def log_llm_call(
    trace: Trace,
    agent_name: str,
    detail: str,
    payload_summary: Optional[dict] = None
):
    """
    Log an LLM call event.

    Args:
        trace: Trace object to add event to
        agent_name: Name of the agent making the call
        detail: Human-readable description
        payload_summary: Optional payload data (will be truncated if large)
    """
    trace.add_event(
        agent_name=agent_name,
        step_type="LLM_CALL",
        detail=detail,
        payload=_truncate_payload(payload_summary)
    )


def log_tool_call(
    trace: Trace,
    tool_name: str,
    input_summary: dict
):
    """
    Log a tool call event.

    Args:
        trace: Trace object to add event to
        tool_name: Name of the tool being called
        input_summary: Input parameters
    """
    trace.add_event(
        agent_name=None,
        step_type="TOOL_CALL",
        detail=f"Calling tool: {tool_name}",
        payload=_truncate_payload(input_summary)
    )


def log_tool_result(
    trace: Trace,
    tool_name: str,
    output_summary: dict
):
    """
    Log a tool result event.

    Args:
        trace: Trace object to add event to
        tool_name: Name of the tool that returned
        output_summary: Output data
    """
    trace.add_event(
        agent_name=None,
        step_type="TOOL_RESULT",
        detail=f"Tool result: {tool_name}",
        payload=_truncate_payload(output_summary)
    )


def log_agent_output(
    trace: Trace,
    agent_name: str,
    output_summary: str
):
    """
    Log an agent output event.

    Args:
        trace: Trace object to add event to
        agent_name: Name of the agent
        output_summary: Summary of the output
    """
    trace.add_event(
        agent_name=agent_name,
        step_type="AGENT_OUTPUT",
        detail=output_summary,
        payload=None
    )


def _truncate_payload(payload: Optional[dict], max_length: int = 500) -> Optional[dict]:
    """
    Truncate payload if it's too large.

    Args:
        payload: Payload dict
        max_length: Max length of serialized payload

    Returns:
        Truncated payload or None
    """
    if not payload:
        return None

    serialized = json.dumps(payload)
    if len(serialized) > max_length:
        return {"_truncated": True, "preview": serialized[:max_length] + "..."}

    return payload


def render_trace_as_jsonl(trace: Trace, path: str = None):
    """
    Render trace as JSONL and append to file.

    Args:
        trace: Trace object to render
        path: Path to JSONL file (defaults to config.TRACE_LOG_PATH)
    """
    if path is None:
        path = str(TRACE_LOG_PATH)

    with open(path, "a", encoding="utf-8") as f:
        f.write(trace.model_dump_json() + "\n")


def render_trace_to_stdout(trace: Trace):
    """
    Render trace as a pretty table to stdout.

    Args:
        trace: Trace object to render
    """
    table = Table(title=f"Trace: {trace.trace_id} (Ticket: {trace.ticket_id})")

    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Agent", style="magenta")
    table.add_column("Step Type", style="green")
    table.add_column("Detail", style="white")

    for event in trace.events:
        # Extract just the time part from ISO timestamp
        time_part = event.timestamp.split("T")[1] if "T" in event.timestamp else event.timestamp

        table.add_row(
            time_part[:12],  # HH:MM:SS.mmm
            event.agent_name or "-",
            event.step_type,
            event.detail[:80] + ("..." if len(event.detail) > 80 else "")
        )

    console.print(table)


def print_trace_summary(trace: Trace):
    """
    Print a concise summary of the trace.

    Args:
        trace: Trace object to summarize
    """
    agents_used = set(e.agent_name for e in trace.events if e.agent_name)
    tool_calls = [e for e in trace.events if e.step_type == "TOOL_CALL"]
    llm_calls = [e for e in trace.events if e.step_type == "LLM_CALL"]

    console.print("\n[bold cyan]Trace Summary[/bold cyan]")
    console.print(f"  Trace ID: {trace.trace_id}")
    console.print(f"  Ticket ID: {trace.ticket_id}")
    console.print(f"  Total Events: {len(trace.events)}")
    console.print(f"  Agents Involved: {', '.join(sorted(agents_used))}")
    console.print(f"  LLM Calls: {len(llm_calls)}")
    console.print(f"  Tool Calls: {len(tool_calls)}")
