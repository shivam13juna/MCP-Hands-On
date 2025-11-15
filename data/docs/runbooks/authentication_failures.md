# Authentication Failures

This runbook helps diagnose login and authentication problems:

Symptoms:
- User cannot sign in; sees **"Invalid credentials"** repeatedly.
- SSO login fails with **"SAML assertion invalid"** or **"OIDC token expired"**.
- API calls return HTTP 401 or 403 unexpectedly.

Resolution steps:
1. Check the status of `auth_service` in the status page.
2. Confirm the user's account is active and not locked.
3. For SSO:
   - Verify SAML / OIDC configuration has not changed.
   - Confirm the user's identity provider is reachable.
4. For API tokens:
   - Ensure the token has not expired or been revoked.
   - Check scopes on the token.

Escalate as a **possible bug** if:
- Multiple users in the same org report the issue simultaneously.
- Status shows `auth_service` as healthy but login errors continue.
