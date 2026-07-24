#!/usr/bin/env python3
"""Register (or update) `embed-generate` and `model-train` as Tapis Actors.

Mirrors this repo's own `tapis/register_pod.py` (argparse, `--dry-run`,
`.env` loading, upsert-via-get/create) and `modflow-suite/subside/tapis/
register_pods.py`'s multi-service-in-one-script shape, adapted for Tapis's
ACTORS API (Abaco) -- a genuinely different subsystem from Pods, not just a
renamed copy. Real facts this script is grounded in:

    1. Registration: `t.actors.create_actor(image=..., name=..., description=...,
       default_environment={...}, stateless=True, hints=[...])` via tapipy.
    2. **CORRECTED, 2026-07-23**: an earlier version of this script required
       a public Docker Hub image, citing Tapis's own readthedocs page
       verbatim ("Abaco pulls images for its actors from the public Docker
       Hub"). Directly corrected by the project owner, who has real,
       hands-on Tapis account experience this session's doc fetches don't
       have (the same readthedocs page was ALSO wrong or incomplete
       elsewhere this session -- e.g. it didn't know about the `17postgis3.5`
       Pod template or the real Pod hostname behavior confirmed in Decision
       33). GHCR works for Actors the same way it already does for Pods
       (`label-studio-tapis-auth`) -- images are built/pushed via
       `.github/workflows/docker-build.yml` (added this increment, mirroring
       that repo's own workflow), not a manual Docker Hub push. Do not
       resurrect the Docker Hub requirement without a real, current source.
    3. Execution: `t.actors.send_message(actor_id=..., request_body={'message':
       <json-string-or-dict>})` queues one execution -- NOT implemented by
       this script (invocation is WebODM's `embeddings_client.py`'s job, see
       that module's own `queue_embed_generate()`/`queue_model_train()`
       stubs) -- this script only registers the Actors.

A real structural difference from `register_pod.py`'s upsert, not a
cosmetic one: Tapis Pods let the CALLER choose `pod_id` up front, so
"does this already exist" is a simple `get_pod(pod_id=...)` call. Tapis's
Actor-creation API returns a server-generated `actor_id` instead -- there is
no caller-chosen identifier to look up by. This script's upsert therefore
searches for an existing actor by `name` via `t.actors.list_actors()`
first -- see `find_existing_actor_id()`'s own docstring for why that
specific method/attribute shape is a judgment call, not one of the facts
confirmed above.

Usage:
    export TAPIS_USERNAME=... TAPIS_PASSWORD=...      # or you'll be prompted
    python tapis/register_actor.py --dry-run
    python tapis/register_actor.py
    python tapis/register_actor.py --actors embed-generate
    python tapis/register_actor.py --recreate

Prerequisites:
    * Push to `main` (or run the workflow manually) so
      `.github/workflows/docker-build.yml` builds + pushes both images to
      GHCR first -- this script does not build or push images itself. Images
      land at `ghcr.io/in-for-disaster-analytics/embeddings-tapis-actors`,
      tagged `embed-generate-latest`/`model-train-latest` (see the workflow's
      matrix). The GHCR package must be set PUBLIC after the first push
      (Abaco pulls anonymously, same requirement as the Pods precedent) --
      `gh api` command for this is in the workflow's own comments.

NOT resolved by this script, stated plainly rather than glossed over
(Decision 30): `default_environment` below is static configuration baked in
at Actor-registration time -- it is NOT a live per-request user credential.
It cannot by itself answer how `embed-generate`/`model-train` authorize
themselves against embeddingsdb/Tapis when Celery queues them
asynchronously (a stored, refreshable service token vs. some other
mechanism) -- that remains the open question Decision 30 describes; this
script only registers the Actors' image/description/env, it does not solve
their runtime credential model.

CAUTION: this script's non-dry-run path has NOT been run against a live
Tapis tenant this session (no credentials were available, and doing so was
explicitly out of scope for this increment -- see this repo's README).
Only `--dry-run` has been exercised. Confirm the resulting spec (--dry-run
first) and the `find_existing_actor_id()`/`update_actor`/`delete_actor`
calls' real behavior against Tapis's actual Actors API before trusting the
non-dry-run path in production.
"""

from __future__ import annotations

import argparse
import json
import os
from getpass import getpass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(override: bool = False) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=override)
    except ImportError:
        pass


# Registry of what each Actor actually is, for spec-building. `env_keys` is
# deliberately narrow: only environment variables the Actor's OWN code
# genuinely calls os.environ.get()/os.environ[...] for TODAY -- not env vars
# it will need once more of it is implemented. Don't invent env vars the
# code doesn't use.
ACTORS = {
    "embed-generate": {
        "description": (
            "Runs Clay v1.5 over a WebODM task's or STAC item's tiles, "
            "writing tile_observations/embeddings/covariates to embeddingsdb. "
            "See design spec Decisions 9, 19, 20, 23, 24, 25, 27, 35."
        ),
        # Decision 44: embed_generate/main.py's run() is now real and calls
        # os.environ.get('WEBODM_URL') directly -- a genuine, newly-added
        # env key, not present when this Actor was first registered
        # (Decision 37, before run() had any real logic). CLAY_CHECKPOINT_PATH/
        # CLAY_MODEL_DIR are deliberately NOT listed here even though the
        # code reads them too -- Decision 39/44 bakes them into the image
        # itself as Dockerfile ENV defaults, so they don't need to be part
        # of the Actor's registered default_environment.
        "env_keys": ["EMBEDDINGSDB_URL", "WEBODM_URL"],
        "image_tag_suffix": "embed-generate",
    },
    "model-train": {
        "description": (
            "Trains a classifier over pooled embeddings/covariates/labels "
            "across any tasks/projects, tracked in MLflow. See design spec "
            "Decisions 16, 17, 25, 28, 35."
        ),
        # Judgment call, stated plainly: model_train/main.py's functions are
        # ALL still NotImplementedError stubs this increment (only its
        # read_actor_message()/main() message-contract fix landed -- see
        # Decision 35) -- there is no os.environ.get() call anywhere in its
        # code yet, for EMBEDDINGSDB_URL, WO_MLFLOW_TRACKING_URI, or
        # anything else. This list is genuinely empty, not an omission --
        # populate it once model_train gets its own DB/MLflow client
        # module(s) that actually consume those variables.
        "env_keys": [],
        "image_tag_suffix": "model-train",
    },
}

DEFAULT_GHCR_IMAGE = "ghcr.io/in-for-disaster-analytics/embeddings-tapis-actors"

# Env vars that may contain secrets (connection strings with embedded
# passwords, etc.) -- masked in --dry-run output, same convention as
# register_pod.py's SECRET_KEYS.
SECRET_KEYS = {"EMBEDDINGSDB_URL"}


def build_actor_spec(name: str, ghcr_image: str, image_tag_suffix_override: str | None = None) -> dict:
    cfg = ACTORS[name]
    tag = image_tag_suffix_override or f"{cfg['image_tag_suffix']}-latest"
    image = f"{ghcr_image}:{tag}"
    default_environment = {k: os.environ[k] for k in cfg["env_keys"] if os.environ.get(k)}
    return {
        "image": image,
        "name": name,
        "description": cfg["description"],
        "default_environment": default_environment,
        "stateless": True,
        # Abaco "hints" request specific runtime capabilities (e.g. GPU
        # scheduling) -- not specified here since neither Actor's current
        # stub code has real compute requirements to hint at yet (no Clay
        # checkpoint loaded, no real training loop running). Revisit once
        # embed-generate's real Clay v1.5 inference is wired in -- confirm
        # against Tapis's own hints vocabulary first, not assumed here.
        "hints": [],
    }


def find_existing_actor_id(t, name: str):
    """
    Look up an existing Actor by `name`, for upsert-style register
    semantics.

    JUDGMENT CALL, not one of this task's confirmed facts: unlike Pods
    (caller-chosen `pod_id`, so `get_pod(pod_id=...)` is a direct
    existence check), Tapis's Actor-creation API returns a server-generated
    `actor_id` -- there is no caller-chosen identifier to look up by
    directly. This function assumes `t.actors.list_actors()` returns
    objects exposing `.id` and `.name` attributes, following the same
    tapipy client-generation convention this repo's `register_pod.py`
    already relies on for Pods (`list_pods`/`get_pod`/etc.) -- but that
    specific method/attribute shape for the ACTORS API has NOT been
    independently verified against a live tenant this session (this
    script's non-dry-run path was explicitly out of scope for this
    increment -- see this repo's README). `--dry-run` never reaches this
    function. Confirm against Tapis's real Actors API (or a real call)
    before trusting it in production.
    """
    for actor in t.actors.list_actors():
        if getattr(actor, "name", None) == name:
            return getattr(actor, "id", None)
    return None


def upsert_actor(t, spec: dict, *, recreate: bool) -> None:
    """
    Create or update one Actor. Same judgment-call caveat as
    `find_existing_actor_id()` above applies to `update_actor`/
    `delete_actor`'s exact call shape -- only `create_actor` and
    `send_message` were independently confirmed against Tapis's own docs
    this increment (see module docstring, facts 1 and 3).
    """
    name = spec["name"]
    existing_id = find_existing_actor_id(t, name)

    if existing_id and recreate:
        print(f"  [{name}] deleting existing actor {existing_id} (--recreate)...")
        t.actors.delete_actor(actor_id=existing_id)
        existing_id = None

    if existing_id:
        print(f"  [{name}] updating actor {existing_id}...")
        t.actors.update_actor(actor_id=existing_id, **spec)
        print(f"  [{name}] actor_id={existing_id} (unchanged by update)")
        return existing_id

    print(f"  [{name}] creating...")
    result = t.actors.create_actor(**spec)
    actor_id = getattr(result, "id", None) or getattr(result, "actor_id", None)
    setting_name = "WO_EMBEDDINGS_ACTOR_ID" if name == "embed-generate" else "WO_MODEL_ACTOR_ID"
    print(f"  [{name}] created actor_id={actor_id}")
    print(f"    -> set {setting_name}={actor_id} in WebODM's settings "
          f"(design spec 'New Django settings') to let embeddings_client.py "
          f"invoke this Actor.")
    return actor_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Register embed-generate and model-train as Tapis Actors.")
    parser.add_argument("--base-url", default=os.environ.get("TAPIS_BASE_URL", "https://portals.tapis.io"))
    parser.add_argument(
        "--ghcr-image", default=DEFAULT_GHCR_IMAGE,
        help="GHCR image (no tag) that .github/workflows/docker-build.yml "
             f"pushes to. Default: {DEFAULT_GHCR_IMAGE}, matching this "
             "project's established GHCR org (in-for-disaster-analytics, "
             "same as label-studio-tapis-auth's Pod image).")
    parser.add_argument("--image-tag-suffix",
                        help="Override the full tag suffix (default: "
                             "'<actor>-latest', matching the workflow's own tags).")
    parser.add_argument("--actors", choices=("both", "embed-generate", "model-train"), default="both")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete + recreate instead of update. NOTE: since actor_id is "
                             "server-generated, --recreate produces a NEW actor_id -- any "
                             "WO_EMBEDDINGS_ACTOR_ID/WO_MODEL_ACTOR_ID already configured in "
                             "WebODM must be updated to match.")
    parser.add_argument("--dry-run", action="store_true", help="Print specs; don't call Tapis.")
    args = parser.parse_args(argv)

    _load_dotenv()
    selected = ["embed-generate", "model-train"] if args.actors == "both" else [args.actors]

    if args.dry_run:
        for name in selected:
            spec = build_actor_spec(name, args.ghcr_image, args.image_tag_suffix)
            masked = dict(spec)
            masked["default_environment"] = {
                k: ("***" if k in SECRET_KEYS else v)
                for k, v in spec["default_environment"].items()
            }
            print(f"--- {name} ---")
            print(json.dumps(masked, indent=2))
        print(
            "\nNOTE (Decision 30, still unresolved by this script): "
            "default_environment above is static config baked in at "
            "registration time -- it is NOT a live per-request user "
            "credential, and this script cannot resolve how "
            "embed-generate/model-train authorize themselves against "
            "embeddingsdb/Tapis once Celery queues them asynchronously "
            "(stored service token vs. some other mechanism)."
        )
        print(
            "\nNOTE: images above must already be pushed to GHCR and set "
            "PUBLIC before real registration can succeed -- push to main (or "
            "run .github/workflows/docker-build.yml manually) first. This "
            "script does not build or push images itself."
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

    for name in selected:
        spec = build_actor_spec(name, args.ghcr_image, args.image_tag_suffix)
        upsert_actor(t, spec, recreate=args.recreate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
