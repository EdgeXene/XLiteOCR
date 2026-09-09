# XLiteOCR deployment requirements

These are preconditions, not suggestions. The code in this repository cannot
establish any of them, and several of the guarantees in `SECURITY.md` and
`app/worker.py` are only true once they are in place. Each section says plainly
what breaks if it is skipped.

Nothing in this file has been applied to a running service. It is written to be
reviewed and then executed by an operator.

## 1. A dedicated unprivileged account

`app/worker.py` gives each document its own process with capped address space,
capped CPU time and no core dumps. That is resource and fault containment. It
is **not** a privilege, filesystem or network boundary: the child runs as the
same user as the parent, with the same access to everything that user can
reach.

The process boundary only limits the blast radius of a memory-safety bug in
PDFium, Pillow, OpenCV, Paddle or VTracer if that user has little to reach.

```bash
useradd --system --home-dir /var/lib/xliteocr --shell /usr/sbin/nologin xliteocr
install -d -o xliteocr -g xliteocr -m 0750 /var/lib/xliteocr
```

Run the service as this account. If it runs as root, a bug reached through a
crafted document is a root compromise, and the containment described in
`app/worker.py` does not apply.

## 2. Models pre-provisioned, read-only

PaddleOCR downloads model weights from the network on first use. A service that
can fetch and execute model files at runtime, from a process that also parses
untrusted documents, has an obvious problem.

Provision the exact artifacts ahead of time and make them read-only to the
service account:

```bash
install -d -o root -g xliteocr -m 0750 /opt/xliteocr/models
# copy the inventoried artifacts into place, then:
chown -R root:xliteocr /opt/xliteocr/models
chmod -R a-w,o-rwx /opt/xliteocr/models
```

Point the service at them and verify before every start. The inventory is
produced alongside the build and is NOT part of this repository, so pass its
path as the argument below. The one measured for the current model set covers
18 files totalling 143,172,277 bytes.

```bash
python - <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path("/opt/xliteocr/models")
inv = json.load(open(sys.argv[1]))   # the inventory JSON for this build
bad = []
for f in inv["files"]:
    p = root / f["path"]
    if not p.exists():
        bad.append(f"missing {f['path']}"); continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
        bad.append(f"hash mismatch {f['path']}")
print("\n".join(bad) if bad else f"{len(inv['files'])} model artifacts verified")
sys.exit(1 if bad else 0)
EOF
```

**These hashes are local measurements, not upstream signatures.** They prove the
files have not changed since they were inventoried on this machine. They do not
prove the files are what PaddlePaddle published. Establishing that requires
comparing against the upstream distribution over a trusted channel, which has
not been done here.

## 3. Restricted network egress

Once the models are provisioned, the service needs no outbound network at all.
Under systemd:

```ini
[Service]
User=xliteocr
Group=xliteocr
# No outbound network. The service listens on loopback and calls nothing.
IPAddressDeny=any
IPAddressAllow=localhost
RestrictAddressFamilies=AF_UNIX AF_INET
# Filesystem
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/xliteocr
PrivateTmp=true
# Kernel and privilege surface
NoNewPrivileges=true
# The child sets RLIMIT_CORE=0 itself, because a core dump of a process holding
# a document is both a denial of service and a copy of that document on disk.
# The PARENT holds the whole upload in memory too, from the moment the body is
# read until the worker answers, so it needs the same limit and cannot set it
# for itself before uvicorn starts.
LimitCORE=0
PrivateDevices=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
MemoryDenyWriteExecute=false   # the model runtimes JIT; this cannot be set
LockPersonality=true
SystemCallArchitectures=native
```

`MemoryDenyWriteExecute` is deliberately left off with its reason recorded: the
inference runtimes generate code at runtime and the service does not start with
it enabled. Do not turn it on without testing that the models still load.

## 4. Temporary files

An upload larger than 1 MB is spooled by the multipart parser to a temporary
file under the system temp directory. It is unlinked at creation, so it has no
name in the filesystem, and its space is reclaimed when the request ends.

On a host whose temp directory is on persistent storage those bytes reach a
disk while the request is in flight. This was measured, not assumed: a 2 MiB
upload produced a descriptor resolving to `/tmp/#61944609 (deleted)` with
`st_size` 2097152, on a filesystem backed by persistent storage.

`PrivateTmp=true` above gives the service its own temp namespace. If document
bytes must never touch persistent storage at all, back it with tmpfs and size
it above the request cap:

```ini
[Service]
Environment=TMPDIR=/run/xliteocr-tmp
```

```ini
[Service]
# systemd creates this under /run, owned by the service user, and removes it
# on stop. Do NOT hand-mount a tmpfs there instead: ProtectSystem=strict in
# section 3 lists /var/lib/xliteocr as the only writable path, so a manual
# mount leaves the service unable to write its own temp directory, and it
# does not survive a reboot.
RuntimeDirectory=xliteocr-tmp
RuntimeDirectoryMode=0700
```

None of these recipes has been run end to end. Test them in staging.

`README.md` and `SECURITY.md` describe the default behavior. If you mount tmpfs,
the pages' current wording remains accurate; it does not claim tmpfs.

### 4a. Reverse-proxy request buffering

If you put a reverse proxy in front of the service, check its request
buffering before you repeat any claim about what stays in memory.

nginx buffers a proxied request body by default, at a compiled default of
8k/16k, into a NAMED temporary file under its client body temp path. That
threshold is far lower than this service's own 1 MB spool, and unlike the
service's file it has a directory entry while it exists. So on a proxied
deployment the proxy's threshold is the one a user actually meets.

To keep the body out of a proxy temp file, in the OCR location only:

```nginx
location /ocr {
    proxy_request_buffering off;
    client_body_buffer_size 1m;
    # keep your existing client_max_body_size and header forwarding
    proxy_pass http://127.0.0.1:3011;
}
```

`proxy_request_buffering off` streams the body through, so nginx writes no
body temp file at all.

Apply it with `nginx -t` followed by `nginx -s reload`. Use a reload, never a
service restart: a reload replaces workers in order and keeps every site on
the host up, while a restart drops live connections.

## 5. Deploy order: engine first, pages second

If you publish pages that describe the service's behavior, deploy the engine
first. This release changes user-visible behavior: documents over the page
limit are refused rather than truncated, every frame of a multipage TIFF is
read, and a second concurrent document is asked to retry. A page deployed
ahead of the engine advertises a guarantee the running service does not yet
have.

## 6. Reload procedure

If you run the service under a process manager, prefer a graceful reload over
a restart so in-flight requests are not dropped.

## 7. What is still not established

- Model artifact hashes are local measurements, not upstream signatures.
- Windows has no POSIX rlimits. The process boundary and wall deadline apply
  there; the memory and CPU caps do not. Do not describe a Windows deployment
  as equivalently contained.
- The worker is not a privilege, filesystem or network sandbox. Sections 1 to 3
  are what make it meaningful, and they are the operator's to apply.
