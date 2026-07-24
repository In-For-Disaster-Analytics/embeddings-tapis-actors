# embeddings-tapis-actors
#
# Image for the two Actors in this repo (embed_generate, model_train).
# embed_generate has real logic as of Decision 44 (tile enumeration/fetch,
# Clay v1.5 inference, embeddingsdb writes -- WebODM-sourced visits only,
# Decision 40); model_train is still a message-contract-only stub (Decision
# 35) -- its own DB/MLflow/scikit-learn un-stubbing is later, separate work.
#
# NOTE on Tapis Actor image conventions -- RESOLVED, see design spec
# Decision 35/36: Tapis's own readthedocs page states verbatim "Abaco pulls
# images for its actors from the public Docker Hub" -- but that page has
# been wrong/incomplete elsewhere this project too (e.g. it didn't know
# about the real `17postgis3.5` Pod template or the real Pod hostname
# behavior, both confirmed by direct experimentation). The project owner,
# with real Tapis account experience, confirmed GHCR works for Actors the
# same way it already does for this repo's sibling label-studio-tapis-auth
# Pod. Images are built/pushed via .github/workflows/docker-build.yml (added
# this increment, mirroring that repo's own workflow) to
# ghcr.io/in-for-disaster-analytics/embeddings-tapis-actors, tagged
# embed-generate-latest/model-train-latest. See tapis/register_actor.py.
#
# Decision 39/44 -- Clay v1.5 baked into the image, not a mounted Tapis
# Volume: `vendor/clay-model-src` (Clay Foundation's own repo, vendored as
# real source -- ~350KB, no large binaries, safe to commit) is COPYed and
# `pip install -e`d below. `vendor/clay-v1.5.ckpt` (~4.8GB) is COPYed too,
# but is NEVER committed to git (`.gitignore`'s `*.ckpt` rule) -- it only
# exists locally, copied in from embeddings-research/models/clay-v1.5.ckpt
# for this Dockerfile to pick up. REAL, FLAGGED GAP: the GHCR CI workflow
# (.github/workflows/docker-build.yml, `context: .`, single-repo checkout)
# has no way to obtain this file -- a real CI build of embed_generate's
# image will FAIL at the checkpoint COPY step until the checkpoint is
# fetched from somewhere CI can reach (a Tapis Files endpoint, a release
# asset, git-lfs, etc. -- not decided here). This Dockerfile is verified to
# build and run correctly LOCALLY, where the checkpoint is physically
# present; the CI path is explicitly not solved by this increment.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Decision 39/44: vendored Clay v1.5 source + checkpoint, embed_generate
# only (model_train's own Dockerfile needs would diverge here once it has
# real training logic -- not done, see this Dockerfile's own top comment).
COPY vendor/clay-model-src ./vendor/clay-model-src
RUN pip install --no-cache-dir -e ./vendor/clay-model-src
COPY vendor/clay-v1.5.ckpt /models/clay-v1.5.ckpt
ENV CLAY_MODEL_DIR=/app/vendor/clay-model-src
ENV CLAY_CHECKPOINT_PATH=/models/clay-v1.5.ckpt

COPY embed_generate ./embed_generate
COPY model_train ./model_train

# Which Actor this image runs -- build with
#   docker build --build-arg ACTOR=embed_generate -t embed-generate .
#   docker build --build-arg ACTOR=model_train    -t model-train .
# A real deployment may instead split this into two separate Dockerfiles
# (Dockerfile.embed-generate / Dockerfile.model-train) once each Actor is
# actually registered with Tapis and their runtime needs diverge further
# (model_train has no real logic to diverge over yet -- see top comment).
ARG ACTOR=embed_generate
ENV ACTOR=${ACTOR}

ENTRYPOINT ["sh", "-c", "python -m ${ACTOR}.main"]
