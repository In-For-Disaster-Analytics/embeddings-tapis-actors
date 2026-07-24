#!/usr/bin/env python3
"""Register (or update) `embed-generate-ls6` as a Tapis App.

Mirrors this repo's own `tapis/register_actor.py` (argparse, `--dry-run`,
`.env` loading) and `tapis/register_pod.py`'s upsert-via-get/create shape --
Apps are structurally closer to Pods than to Actors for this purpose: an App
has a CALLER-CHOSEN `id` (`ls6/app.json`'s own `"id": "embed-generate-ls6"`),
so "does this already exist" is a direct `t.apps.getApp(appId=...)` lookup,
unlike Actors' server-generated `actor_id` (which needed `register_actor.py`'s
own `find_existing_actor_id()` workaround, itself flagged there as an
unverified judgment call).

Real facts this script is grounded in, confirmed against the actually-
installed `tapipy` package this session (not assumed from docs):
    - `t.apps.createAppVersion(**spec)` is the real registration call --
      confirmed via `op_desc.request_body.content['application/json']
      .schema.properties`, a non-empty dict (`id`, `version`, `runtime`,
      `containerImage`, `jobAttributes`, ...), meaning tapipy's generic
      resource dispatcher maps each top-level JSON key to its own kwarg
      (the same rule Decision 43 uncovered for `t.actors.sendMessage`) --
      NOT a single `request_body=` dict (that convenience kwarg only
      applies when a schema has no declared properties, which this one
      does).
    - `t.apps.getApp(appId=..., appVersion=...)` for the existence check --
      confirmed via its real `path_parameters`.
    - `t.jobs.submitJob(**spec)` (used by `embeddings_client.py`'s
      `apply_embed_generate()`, not this script) follows the identical
      non-empty-properties rule.

See `ls6/app.json` for the real App spec this script loads and registers,
and `ls6/tapisjob_app.sh` for what Tapis actually executes on the ls6
compute node once a job runs. Design spec Decision 45
(WebODM/docs/design/2026-07-22-geospatial-embeddings-classification.md)
records why this exists: the embed-generate Actor (Abaco) cannot provision
a worker for this workload's image size on this tenant.

CAUTION, stated plainly rather than assumed: this script's non-dry-run path
has NOT been run against a live Tapis tenant as of writing -- only
`--dry-run` has been exercised. The App's `containerImage` field points at
a ZIP file path on the `cloud.data` Tapis storage system
(`tapis://cloud.data/.../embed-generate-ls6.zip`) that must actually be
uploaded there first (build a ZIP of this directory's `app.json` +
`tapisjob_app.sh`, then SCP/SFTP it to the path in `ls6/app.json`'s own
`containerImage` field, mirroring `nodeodm-ls6/README.md`'s own documented
upload step) -- this script does not build or upload that ZIP itself.

Usage:
    export TAPIS_USERNAME=... TAPIS_PASSWORD=...      # or you'll be prompted
    python tapis/register_app.py --dry-run
    python tapis/register_app.py
"""

from __future__ import annotations

import argparse
import json
import os
from getpass import getpass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JSON_PATH = REPO_ROOT / "ls6" / "app.json"

# Env vars that may contain secrets -- masked in --dry-run output, same
# convention as register_actor.py's/register_pod.py's SECRET_KEYS.
SECRET_KEYS = {"EMBEDDINGSDB_URL", "MSG"}


def _load_dotenv(override: bool = False) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=override)
    except ImportError:
        pass


def load_app_spec() -> dict:
    with open(APP_JSON_PATH) as f:
        return json.load(f)


def mask_secrets(spec: dict) -> dict:
    masked = json.loads(json.dumps(spec))  # deep copy
    env_vars = masked.get("jobAttributes", {}).get("parameterSet", {}).get("envVariables", [])
    for ev in env_vars:
        if ev.get("key") in SECRET_KEYS and ev.get("value"):
            ev["value"] = "***"
    return masked


def upsert_app(t, spec: dict) -> None:
    """
    Create or update the App. Unlike `register_actor.py`'s Actor upsert
    (server-generated id, requires a name-based search workaround), Apps
    have a caller-chosen `id` (`spec['id']`) -- existence is a direct
    `getApp()` call, matching `register_pod.py`'s own Pod-upsert pattern.
    """
    app_id = spec["id"]
    app_version = spec["version"]

    try:
        existing = t.apps.getApp(appId=app_id)
        print(f"  [{app_id}] already exists (latest version {getattr(existing, 'version', '?')}) "
              f"-- creating version {app_version} as a new version.")
    except Exception:
        print(f"  [{app_id}] does not exist yet -- creating version {app_version}.")

    result = t.apps.createAppVersion(**spec)
    print(f"  [{app_id}] createAppVersion succeeded: {result}")
    print(f"    -> set an app-id setting (e.g. WO_EMBED_GENERATE_APP_ID={app_id}) "
          f"in WebODM's settings so embeddings_client.py's apply_embed_generate() "
          f"can submit jobs against it (see design spec Decision 45).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Register embed-generate-ls6 as a Tapis App.")
    parser.add_argument("--base-url", default=os.environ.get("TAPIS_BASE_URL", "https://portals.tapis.io"))
    parser.add_argument("--dry-run", action="store_true", help="Print the spec; don't call Tapis.")
    args = parser.parse_args(argv)

    _load_dotenv()
    spec = load_app_spec()

    if args.dry_run:
        print(json.dumps(mask_secrets(spec), indent=2))
        print(
            "\nNOTE: containerImage above points at a ZIP on the cloud.data "
            "Tapis storage system that must be uploaded there first (build "
            "a ZIP of ls6/app.json + ls6/tapisjob_app.sh, then SCP/SFTP it "
            "to that path) -- this script does not build or upload it."
        )
        return 0

    try:
        from tapipy.tapis import Tapis
    except ImportError:
        raise SystemExit("tapipy is not installed (pip install tapipy).")

    username = os.environ.get("TAPIS_USERNAME") or input("Tapis username: ")
    password = os.environ.get("TAPIS_PASSWORD") or getpass("Tapis password: ")
    t = Tapis(base_url=args.base_url.rstrip("/"), username=username, password=password)
    t.get_tokens()

    upsert_app(t, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
