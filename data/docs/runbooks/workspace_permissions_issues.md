# Workspace Permissions Issues

This runbook covers common permission-related problems in workspaces:

Symptoms:
- User can see the workspace but cannot view certain dashboards.
- User cannot edit dashboards even though they should be an editor.
- Sharing links show **"You don't have access"** errors.

Resolution steps:
1. Ask the user for the workspace name and their role.
2. In the admin console, verify the user's role is `viewer`, `editor`, or `admin`
   as expected.
3. If the user belongs to a group, check group-level overrides.
4. For "cannot edit" issues, confirm the dashboard owner has not locked editing.
5. If permissions look correct but behavior does not match, search for incidents
   tagged with `workspace` and `permissions`.

Notes:
- Many permission issues are due to group overrides.
- Always verify group membership when in doubt.
