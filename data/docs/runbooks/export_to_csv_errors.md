# Export to CSV Errors

When users attempt to export reports to CSV and see errors like
**"Export failed: timeout"** or **"Export failed: invalid workspace"**, follow
these steps:

1. Confirm the user has the `export_reports` permission on the workspace.
2. Check recent incidents related to the `export_service`.
3. Ask the user to try a smaller date range if they are exporting very large data.
4. If the issue persists and no incident is active, escalate as a **possible bug**
   with the error ID shown in logs.

Common causes:
- Workspace permission misconfiguration.
- Temporary outages of the export pipeline.
- Extremely large exports hitting timeouts.
