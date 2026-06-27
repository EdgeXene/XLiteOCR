# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in XLiteOCR, please report it
privately. Do not open a public issue for security-sensitive reports.

Email: **security@edgexene.io**

Please include:

- A description of the issue and its impact.
- Steps to reproduce, or a proof of concept.
- The affected version or commit.

We will acknowledge your report and work with you on a fix and coordinated
disclosure.

## Scope and deployment notes

XLiteOCR is a self-hosted HTTP service. A few operational notes that affect its
security posture:

- **It binds to `127.0.0.1` by default.** It is not hardened for direct exposure
  to the public internet. Place it behind a reverse proxy with TLS,
  authentication, and request-size limits if you expose it beyond localhost.
- **Uploaded documents are processed in memory and are not written to disk** by
  the service itself. Your surrounding infrastructure (reverse proxy logs, etc.)
  may still record request metadata.
- **No authentication is built in.** Access control is intentionally left to your
  deployment (reverse proxy, network policy, or an auth gateway in front).
- Model weights are downloaded from the upstream PaddleOCR distribution on first
  run. Pin and mirror them internally for air-gapped or reproducible builds.
