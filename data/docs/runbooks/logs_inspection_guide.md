# Logs Inspection Guide

Use this guide when a ticket includes a log snippet:

1. Look for error codes (e.g., `ERR_EXPORT_TIMEOUT`, `ERR_AUTH_INVALID_TOKEN`).
2. Match error codes to known runbooks such as **Export to CSV Errors** or
   **Authentication Failures**.
3. Pay attention to correlation IDs and timestamps; these may be needed if you
   escalate to engineering.
4. If logs mention an internal microservice name (like `export_service`), check
   its status and recent incidents.
