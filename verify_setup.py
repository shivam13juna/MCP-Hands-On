"""
Verification script to check if the project is set up correctly.
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and return result."""
    exists = path.exists()
    return exists


def check_directory_exists(path: Path, description: str) -> bool:
    """Check if a directory exists and return result."""
    exists = path.exists() and path.is_dir()
    return exists


def main():
    """Run verification checks."""
    console.print("\n[bold cyan]AI Support Copilot - Setup Verification[/bold cyan]\n")

    base_dir = Path(__file__).parent
    all_checks = []

    # Check data files
    checks = [
        ("Data Directories", [
            (base_dir / "data" / "docs" / "runbooks", "Runbooks directory", check_directory_exists),
            (base_dir / "data" / "incidents", "Incidents directory", check_directory_exists),
            (base_dir / "data" / "status", "Status directory", check_directory_exists),
            (base_dir / "data" / "tickets", "Tickets directory", check_directory_exists),
        ]),
        ("Runbooks", [
            (base_dir / "data" / "docs" / "runbooks" / "export_to_csv_errors.md", "Export runbook", check_file_exists),
            (base_dir / "data" / "docs" / "runbooks" / "authentication_failures.md", "Auth runbook", check_file_exists),
            (base_dir / "data" / "docs" / "runbooks" / "workspace_permissions_issues.md", "Permissions runbook", check_file_exists),
        ]),
        ("Data Files", [
            (base_dir / "data" / "incidents" / "incidents.json", "Incidents data", check_file_exists),
            (base_dir / "data" / "status" / "status.json", "Status data", check_file_exists),
            (base_dir / "data" / "tickets" / "samples.json", "Sample tickets", check_file_exists),
        ]),
        ("MCP Server", [
            (base_dir / "support_mcp_server" / "config.py", "Server config", check_file_exists),
            (base_dir / "support_mcp_server" / "server.py", "Server main", check_file_exists),
            (base_dir / "support_mcp_server" / "tools_support_docs.py", "Docs tool", check_file_exists),
            (base_dir / "support_mcp_server" / "tools_incidents.py", "Incidents tool", check_file_exists),
            (base_dir / "support_mcp_server" / "tools_status.py", "Status tool", check_file_exists),
        ]),
        ("Host Application", [
            (base_dir / "support_copilot_host" / "config.py", "Host config", check_file_exists),
            (base_dir / "support_copilot_host" / "models.py", "Data models", check_file_exists),
            (base_dir / "support_copilot_host" / "llm_client.py", "LLM client", check_file_exists),
            (base_dir / "support_copilot_host" / "mcp_client.py", "MCP client", check_file_exists),
            (base_dir / "support_copilot_host" / "agents.py", "Agents", check_file_exists),
            (base_dir / "support_copilot_host" / "orchestrator.py", "Orchestrator", check_file_exists),
            (base_dir / "support_copilot_host" / "cli.py", "CLI", check_file_exists),
        ]),
        ("Documentation", [
            (base_dir / "README.md", "README", check_file_exists),
            (base_dir / "requirements.txt", "Requirements", check_file_exists),
        ])
    ]

    # Run checks
    table = Table(title="Setup Verification Results")
    table.add_column("Category", style="cyan")
    table.add_column("Item", style="white")
    table.add_column("Status", style="green")

    total_checks = 0
    passed_checks = 0

    for category, items in checks:
        for path, description, check_func in items:
            total_checks += 1
            result = check_func(path, description)
            if result:
                passed_checks += 1
                status = "[green]✓ OK[/green]"
            else:
                status = "[red]✗ MISSING[/red]"

            table.add_row(category, description, status)
            # Clear category name for subsequent rows
            category = ""

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {passed_checks}/{total_checks} checks passed")

    # Environment variable check
    import os
    console.print("\n[bold cyan]Environment Variables:[/bold cyan]")
    if os.getenv("OPENAI_API_KEY"):
        console.print("  [green]✓[/green] OPENAI_API_KEY is set")
    else:
        console.print("  [yellow]![/yellow] OPENAI_API_KEY is not set (required for LLM calls)")
        console.print("    Set it with: export OPENAI_API_KEY='your-key'")

    # Final recommendation
    if passed_checks == total_checks:
        console.print("\n[bold green]✓ All checks passed! You're ready to go.[/bold green]")
        console.print("\nNext steps:")
        console.print("  1. Set OPENAI_API_KEY if not already set")
        console.print("  2. Install dependencies: pip install -r requirements.txt")
        console.print("  3. Start MCP server: python -m support_mcp_server.server")
        console.print("  4. Run a sample ticket: python -m support_copilot_host.cli run-sample --id TICKET-001")
        return 0
    else:
        console.print("\n[bold red]✗ Some checks failed. Please review the output above.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
