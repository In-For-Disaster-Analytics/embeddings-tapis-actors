#!/usr/bin/env python3
"""Register (or update) embeddingsdb as a Postgres+PostGIS+pgvector Tapis Pod.

SUPERSEDED BY REALITY, 2026-07-23 -- read this before trusting anything below.
The live embeddingsdb Pod was NOT created by an earlier version of this
script. It was created directly from Tapis's own "17postgis3.5" Pod
TEMPLATE (Postgres 17 + PostGIS 3.5 pre-installed), then pgvector was added
to the *running* container afterward via `tapis/install_pgvector.py`
(`apt-get install postgresql-17-pgvector` through Tapis's real
`exec_pod_commands` API, confirmed to have root/apt access -- not guaranteed
by Tapis's docs, verified empirically). Schema was applied via a plain
`psql -f schema/embeddingsdb.sql` afterward. All of that is real and already
done -- see this repo's README "Status" section.

This script has been rewritten to match that real path (`template=`, not a
custom `image=`) so it stays useful if the Pod ever needs to be recreated
from scratch -- rather than leaving the original, untested plan (a custom
`pgvector/pgvector:pg16` image) as stale, wrong guidance in the repo. Two
real differences from the label-studio-tapis-auth/tapis/register_pod.py
precedent this was adapted from, not just cosmetic renames:

  1. embeddingsdb uses Tapis's own Pod template ("17postgis3.5") -- no
     custom Dockerfile, no GHCR build/push step. pgvector is NOT part of
     that template (confirmed live: `CREATE EXTENSION vector` fails with
     "could not open extension control file" until installed) -- run
     `tapis/install_pgvector.py` once after this script creates/updates
     the Pod.
  2. embeddingsdb is a database, not an HTTP service with a login flow --
     there is no OAuth client to register (that concern is specific to
     Label Studio's Tapis SSO use case and does not apply here; it has been
     removed entirely, not just made optional). Networking uses Tapis's
     "postgres" protocol/port block, not "default"/"http" -- though see
     pod_hostname()'s docstring for a real, unexplained discrepancy between
     that and the actual live Pod's hostname.

This script creates/updates the POD ONLY -- it does NOT install pgvector or
apply the schema. See tapis/install_pgvector.py, schema/embeddingsdb.sql,
and this repo's README "How a future implementer should proceed" for the
full real sequence.

Usage:
    export TAPIS_USERNAME=... TAPIS_PASSWORD=...      # or you'll be prompted
    python tapis/register_pod.py
    python tapis/register_pod.py --recreate             # delete + recreate
    python tapis/register_pod.py --dry-run              # print specs, don't call Tapis
    python tapis/register_pod.py --print-connection-string  # just print EMBEDDINGSDB_URL, no Tapis call

Prerequisites:
    * A Tapis account with Pods access. No image to build/push -- the
      template provides Postgres+PostGIS; pgvector is added post-creation.

CAUTION: this rewritten version has NOT itself been run against Tapis --
only the manually-created Pod (via the Tapis console, not this script) and
tapis/install_pgvector.py have been live-verified. If you use this script
to recreate the Pod, confirm the resulting spec (--dry-run first) actually
matches a "17postgis3.5"-template Pod before trusting it, and re-run
install_pgvector.py + the schema afterward -- a freshly (re)created Pod
starts with neither.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from getpass import getpass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tapis pod_id/volume_id must be lowercase alphanumeric, first char alpha --
# NO hyphens (same constraint label-studio-tapis-auth/tapis/register_pod.py
# notes for pod_id, confirmed the hard way there; applies to volume_id too).
# "embeddingsdb" / "embeddingsdbdata" both satisfy this.
POD_ID = "embeddingsdb"
VOLUME_ID = "embeddingsdbdata"

# Tapis's own built-in Pod template -- Postgres 17 + PostGIS 3.5 -- confirmed
# real by the live Pod (created via the Tapis console, matched here). pgvector
# is NOT part of this template; see tapis/install_pgvector.py, run once after
# this script creates/updates the Pod.
DEFAULT_TEMPLATE = "17postgis3.5"

DEFAULT_POSTGRES_DB = "embeddingsdb"
DEFAULT_POSTGRES_USER = "embeddingsdb"

# Volume sizing: embeddingsdb holds pgvector embeddings (1024-dim float4
# vectors, ~4KB/row for the vector alone -- see schema/embeddingsdb.sql's
# comment on embeddings.vector) plus covariates/labels/predictions rows and
# their indexes, at a scale meant to pool across many sites/tasks/projects
# (Decision 6) -- not a single-container SQLite-scale workload the way
# Label Studio's 10GB default assumed. 50GB is a starting default, not a
# calculated ceiling: pick a bigger number up front rather than resizing a
# live Tapis Volume under a running Postgres instance later.
DEFAULT_VOLUME_SIZE_MB = 51200  # 50 GiB

SECRET_KEYS = {"POSTGRES_PASSWORD"}


def _load_dotenv(override: bool = False) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=override)
    except ImportError:
        pass


def _pods_domain(base_url: str) -> str:
    return base_url.rstrip("/").split("://", 1)[-1]


def pod_hostname(base_url: str) -> str:
    """External hostname for embeddingsdb.

    CORRECTED against the real, live Pod, 2026-07-23 -- an earlier version of
    this function predicted a "-postgres" suffix (reasoning: a non-"default"
    networking key allocates a separate subdomain interface, per Tapis's own
    Pods docs). The actual live Pod's hostname has NO such suffix
    ("embeddingsdb.pods.portals.tapis.io", confirmed via a real `psql`
    connection) -- a bare "{pod_id}.pods.{domain}", same shape as
    label-studio-tapis-auth's single "default" HTTP interface. Why a
    template-created Pod's postgres-protocol interface doesn't get the
    suffix generic Tapis docs implied is not resolved here -- this function
    now matches OBSERVED reality rather than docs-derived prediction, but
    that gap in understanding is worth resolving if it ever matters (e.g.
    a second networking interface on the same Pod).
    """
    return f"{POD_ID}.pods.{_pods_domain(base_url)}"


def connection_string(base_url: str, user: str, password: str, db: str) -> str:
    """Build the real EMBEDDINGSDB_URL shape, matching .env.example.

    Externally, Tapis Pods traffic is reached over port 443 with TLS
    (confirmed against Tapis's own docs) -- the pod's declared networking
    port (5432 below, in build_spec()) is the INTERNAL container port
    Postgres actually listens on; Tapis's own ingress is what's reachable
    from outside, always on 443, regardless of the wrapped protocol. A
    client connecting to this URL is going through that TLS-terminating
    proxy, not talking to port 5432 directly from outside the cluster.
    """
    host = pod_hostname(base_url)
    return f"postgresql://{user}:{password}@{host}:443/{db}?sslmode=require"


def _ensure_postgres_password(env_path: Path) -> str:
    """Return POSTGRES_PASSWORD from the environment, or generate + persist one.

    Mirrors how label-studio-tapis-auth's register_pod.py treats
    TAPIS_CLIENT_SECRET as a sensitive value: never printed in --dry-run
    output (see SECRET_KEYS), and written to .env so re-running this script
    is idempotent rather than rotating the password on every invocation.
    """
    existing = os.environ.get("POSTGRES_PASSWORD")
    if existing:
        return existing

    generated = secrets.token_urlsafe(24)
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    lines = [ln for ln in lines if not ln.startswith("POSTGRES_PASSWORD=")]
    lines.append(f"POSTGRES_PASSWORD={generated}")
    env_path.write_text("\n".join(lines) + "\n")
    os.environ["POSTGRES_PASSWORD"] = generated
    print(f"Generated POSTGRES_PASSWORD and wrote it to {env_path}.")
    return generated


def build_spec(template: str, base_url: str, *, postgres_db: str, postgres_user: str,
               postgres_password: str) -> dict:
    env = {
        "POSTGRES_DB": postgres_db,
        "POSTGRES_USER": postgres_user,
        "POSTGRES_PASSWORD": postgres_password,
    }

    return {
        "pod_id": POD_ID,
        "template": template,
        "description": "embeddingsdb -- Postgres+PostGIS+pgvector store for the "
                        "geospatial embeddings/classification system (Decision 26). "
                        "pgvector added post-creation via install_pgvector.py -- "
                        "not part of the template itself.",
        # "postgres" (not "default"/"http") -- a non-HTTP service on its own
        # subdomain interface, per Tapis's own Pods networking docs. Port
        # 5432 here is the INTERNAL container port Postgres listens on;
        # see connection_string()'s docstring for the external port (443).
        "networking": {"postgres": {"protocol": "postgres", "port": 5432}},
        "resources": {"cpu_request": 500, "cpu_limit": 4000,
                      "mem_request": 1024, "mem_limit": 8192},
        "volume_mounts": {"/var/lib/postgresql/data": {"type": "tapisvolume", "source_id": VOLUME_ID}},
        "environment_variables": env,
        "time_to_stop_default": -1,  # long-running service
    }


def ensure_volume(t, *, size_limit_mb: int, dry_run: bool) -> None:
    if dry_run:
        print(f"Would ensure volume {VOLUME_ID!r} exists (size_limit={size_limit_mb}).")
        return
    try:
        t.pods.get_volume(volume_id=VOLUME_ID)
        print(f"Volume {VOLUME_ID!r} already exists.")
    except Exception:
        print(f"Creating volume {VOLUME_ID!r}...")
        t.pods.create_volume(
            volume_id=VOLUME_ID,
            description="embeddingsdb Postgres data directory (/var/lib/postgresql/data)",
            size_limit=size_limit_mb,
        )


def upsert_pod(t, spec: dict, *, recreate: bool, start: bool, restart: bool) -> None:
    pid = spec["pod_id"]
    exists = True
    try:
        t.pods.get_pod(pod_id=pid)
    except Exception:
        exists = False

    if exists and recreate:
        print(f"  [{pid}] deleting existing pod (--recreate)...")
        t.pods.delete_pod(pod_id=pid)
        exists = False

    if exists:
        print(f"  [{pid}] updating...")
        t.pods.update_pod(**spec)
        if restart:
            try:
                t.pods.restart_pod(pod_id=pid)
                print(f"  [{pid}] restart requested (applying env changes)")
            except Exception as exc:
                print(f"  [{pid}] restart failed: {exc}")
            return
    else:
        print(f"  [{pid}] creating...")
        t.pods.create_pod(**spec)

    if start:
        try:
            status = getattr(t.pods.get_pod(pod_id=pid), "status", None)
        except Exception:
            status = None
        if status and status != "STOPPED":
            print(f"  [{pid}] already {status}; not starting")
        else:
            try:
                t.pods.start_pod(pod_id=pid)
                print(f"  [{pid}] start requested")
            except Exception as exc:
                print(f"  [{pid}] start skipped: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Register embeddingsdb as a Tapis Pod (Postgres+pgvector).")
    parser.add_argument("--base-url", default=os.environ.get("TAPIS_BASE_URL", "https://portals.tapis.io"))
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help="Tapis Pod template (Postgres+PostGIS). pgvector is added "
                             "separately -- see tapis/install_pgvector.py.")
    parser.add_argument("--postgres-db", default=os.environ.get("POSTGRES_DB", DEFAULT_POSTGRES_DB))
    parser.add_argument("--postgres-user", default=os.environ.get("POSTGRES_USER", DEFAULT_POSTGRES_USER))
    parser.add_argument("--volume-size-mb", type=int, default=DEFAULT_VOLUME_SIZE_MB)
    parser.add_argument("--recreate", action="store_true", help="Delete + recreate instead of update.")
    parser.add_argument("--restart", action="store_true",
                        help="Restart the updated pod so env changes take effect.")
    parser.add_argument("--no-start", action="store_true", help="Create/update but don't start.")
    parser.add_argument("--dry-run", action="store_true", help="Print specs; don't call Tapis.")
    parser.add_argument("--print-connection-string", action="store_true",
                        help="Just print the EMBEDDINGSDB_URL shape and exit; don't call Tapis.")
    args = parser.parse_args(argv)

    _load_dotenv()
    env_path = REPO_ROOT / ".env"

    if args.print_connection_string:
        password = os.environ.get("POSTGRES_PASSWORD", "<POSTGRES_PASSWORD>")
        url = connection_string(args.base_url, args.postgres_user,
                                 "***" if password != "<POSTGRES_PASSWORD>" else password,
                                 args.postgres_db)
        print(url)
        return 0

    if args.dry_run:
        # No password generation/mutation in --dry-run -- print the spec
        # shape without touching .env or requiring a real secret yet.
        placeholder_password = os.environ.get("POSTGRES_PASSWORD", "<generated-if-unset>")
        spec = build_spec(args.template, args.base_url,
                           postgres_db=args.postgres_db,
                           postgres_user=args.postgres_user,
                           postgres_password=placeholder_password)
        spec["environment_variables"] = {
            k: ("***" if k in SECRET_KEYS else v)
            for k, v in spec["environment_variables"].items()
        }
        print(json.dumps(spec, indent=2))
        print(f"\nEMBEDDINGSDB_URL once running: "
              f"{connection_string(args.base_url, args.postgres_user, '***', args.postgres_db)}")
        ensure_volume(None, size_limit_mb=args.volume_size_mb, dry_run=True)
        return 0

    try:
        from tapipy.tapis import Tapis
    except ImportError:
        raise SystemExit("tapipy is not installed (pip install tapipy).")

    username = os.environ.get("TAPIS_USERNAME") or input("Tapis username: ")
    password = os.environ.get("TAPIS_PASSWORD") or getpass("Tapis password: ")
    t = Tapis(base_url=args.base_url.rstrip("/"), username=username, password=password)
    t.get_tokens()

    postgres_password = _ensure_postgres_password(env_path)

    ensure_volume(t, size_limit_mb=args.volume_size_mb, dry_run=False)

    spec = build_spec(args.template, args.base_url,
                       postgres_db=args.postgres_db,
                       postgres_user=args.postgres_user,
                       postgres_password=postgres_password)

    print("WARNING: POSTGRES_PASSWORD will be stored in the pod's environment_variables "
          "(visible to the pod owner). For production, move it to Tapis secrets "
          "(${pods:secrets:POSTGRES_PASSWORD}) instead.\n")

    upsert_pod(t, spec, recreate=args.recreate, start=not args.no_start, restart=args.restart)

    url = connection_string(args.base_url, args.postgres_user, postgres_password, args.postgres_db)
    print(f"\nDone. Pod (once started) reachable at: {pod_hostname(args.base_url)}")
    print(f"EMBEDDINGSDB_URL={url}")
    print("\nThis creates the Pod ONLY -- it does not apply the schema. Once the pod "
          "is reachable, apply it separately:\n"
          "  psql \"$EMBEDDINGSDB_URL\" -f schema/embeddingsdb.sql")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
