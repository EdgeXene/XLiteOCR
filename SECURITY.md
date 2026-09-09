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
- **Uploaded documents are not retained.** An upload exists only for the
  request that sent it. Uploads up to 1 MB are held in memory. Above that, the
  multipart parser spools the body to a temporary file under the system temp
  directory; it is unlinked on creation, so it has no name in the filesystem,
  and its space is reclaimed when the request ends. On a host where the temp
  directory is on persistent storage, those bytes do reach a disk while the
  request is in flight. Mount the temp directory on tmpfs if that matters for
  your threat model. Nothing is written to a database or any lasting location.
  A reverse proxy in front of the service has its own request buffering, with
  its own threshold (nginx defaults to 8k/16k) and its own named temporary file
  under its body temp path. That layer is outside this service's control and is
  covered in DEPLOYMENT.md. Your surrounding infrastructure (reverse proxy logs,
  etc.) may still record request metadata.
- **Worker error messages are withheld by default.** Each document is processed
  in a child process. When it fails, only the exception class name is logged,
  because a decoder's exception message routinely quotes the bytes that upset
  it and the service log is a file on persistent storage. Setting
  `XLITE_DEBUG_WORKER_ERRORS=1` prints the full message on the worker's own
  stderr instead. Under the service that stream is drained into a small
  in-memory ring and is never logged, so the switch is observable mainly when
  `python -m app.worker` is run directly; it can still place document content
  anywhere that stderr is captured. It is off by default and should stay off in
  production.
- **No authentication is built in.** Access control is intentionally left to your
  deployment (reverse proxy, network policy, or an auth gateway in front).
- Model weights are downloaded from the upstream PaddleOCR distribution on first
  run. Pin and mirror them internally for air-gapped or reproducible builds.
