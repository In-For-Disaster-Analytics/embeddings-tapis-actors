#!/usr/bin/env python3
"""Install the pgvector extension into the live embeddingsdb Pod, in place.

Context: embeddingsdb was created directly from Tapis's own "17postgis3.5"
Pod template (Postgres 17 + PostGIS 3.5 pre-installed) rather than via this
repo's register_pod.py (which had assumed a custom pgvector/pgvector:pg16
image -- superseded by this real choice). Confirmed live against the actual
running Pod (`psql`, 2026-07-23): PostGIS 3.5.2 works; pgvector is genuinely
absent ("could not open extension control file ... vector.control: No such
file or directory") -- not just un-created, the extension files aren't there.

Verified locally (Docker, 2026-07-23) that this is fixable without rebuilding
the Pod's image: `postgis/postgis:17-3.5` (the public image this Pod's
template is presumably built from) + `apt-get install postgresql-17-pgvector`
+ `CREATE EXTENSION vector` works end-to-end, including a real vector
distance query. This script runs that same apt-get install command inside
the LIVE Pod via Tapis's own `exec_pod_commands` API, then leaves the
`CREATE EXTENSION vector` step to a plain psql connection (see main()) --
that part needs no special Tapis permissions, just the existing
EMBEDDINGSDB_URL credentials already in .env.

UNVERIFIED: whether exec_pod_commands has root/apt/network access inside
this specific container. If this script fails with a permissions error
(not a "command not found" or network error), that's the real answer, and
the fallback is a custom-built image (postgis/postgis:17-3.5 + pgvector
baked in at build time, already verified working) via register_pod.py's
`image=` path, recreating the Pod -- see README.

Usage:
    export TAPIS_USERNAME=... TAPIS_PASSWORD=...      # or you'll be prompted
    python tapis/install_pgvector.py --pod-id embeddingsdb
    python tapis/install_pgvector.py --dry-run          # print commands, don't call Tapis
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

APT_COMMANDS = [
    ["apt-get", "update"],
    ["apt-get", "install", "-y", "--no-install-recommends", "postgresql-17-pgvector"],
]


def _load_dotenv(override: bool = False) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=override)
    except ImportError:
        pass


def create_extension_via_psql() -> int:
    """CREATE EXTENSION vector over a plain psql connection -- no Tapis
    credentials needed here, just EMBEDDINGSDB_URL (already in .env).
    Requires the `psql` client to be on PATH.
    """
    url = os.environ.get("EMBEDDINGSDB_URL")
    if not url:
        print("EMBEDDINGSDB_URL is not set (check .env) -- cannot run CREATE EXTENSION.",
              file=sys.stderr)
        return 1
    print("Running CREATE EXTENSION vector over psql...")
    result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-c",
         "CREATE EXTENSION IF NOT EXISTS vector; "
         "SELECT extversion FROM pg_extension WHERE extname = 'vector';"],
    )
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Install pgvector into the live embeddingsdb Pod via exec_pod_commands.")
    parser.add_argument("--base-url", default=os.environ.get("TAPIS_BASE_URL", "https://portals.tapis.io"))
    parser.add_argument("--pod-id", default=os.environ.get("EMBEDDINGSDB_POD_ID", "embeddingsdb"),
                        help="Real pod_id of the already-created Pod (confirm this matches "
                             "what you see in the Tapis console -- do not assume).")
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Print commands; don't call Tapis.")
    parser.add_argument("--skip-create-extension", action="store_true",
                        help="Only run apt-get inside the Pod; skip the psql CREATE EXTENSION step.")
    args = parser.parse_args(argv)

    _load_dotenv()

    if args.dry_run:
        print(f"Would exec_pod_commands(pod_id={args.pod_id!r}, commands={APT_COMMANDS!r})")
        print("Would then run: psql $EMBEDDINGSDB_URL -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        return 0

    try:
        from tapipy.tapis import Tapis
    except ImportError:
        raise SystemExit("tapipy is not installed (pip install tapipy).")

    username = os.environ.get("TAPIS_USERNAME") or input("Tapis username: ")
    password = os.environ.get("TAPIS_PASSWORD") or getpass("Tapis password: ")
    t = Tapis(base_url=args.base_url.rstrip("/"), username=username, password=password)
    t.get_tokens()

    print(f"Running {len(APT_COMMANDS)} commands inside Pod {args.pod_id!r}...")
    try:
        response = t.pods.exec_pod_commands(
            pod_id=args.pod_id,
            commands=APT_COMMANDS,
            command_timeout=args.command_timeout,
            total_timeout=args.command_timeout,
            fail_on_non_success=True,
        )
        print(response)
    except Exception as exc:
        print(f"exec_pod_commands failed: {exc}", file=sys.stderr)
        print("If this is a permissions error (not 'command not found' or a network "
              "error), the container likely doesn't allow apt/root access via exec -- "
              "see this script's module docstring for the fallback (custom image, "
              "Pod recreation).", file=sys.stderr)
        return 1

    if args.skip_create_extension:
        print("Skipping CREATE EXTENSION per --skip-create-extension.")
        return 0

    return create_extension_via_psql()


if __name__ == "__main__":
    raise SystemExit(main())
