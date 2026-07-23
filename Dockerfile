# embeddings-tapis-actors
#
# Minimal image for the two Actors in this repo (embed_generate, model_train).
# Neither Actor has real logic yet -- see README "Status: skeleton only" --
# so this Dockerfile only needs to prove the image builds and can run either
# entrypoint; it is not tuned for the real Clay v1.5 inference workload yet
# (no GPU base image, no vendored claymodel/checkpoint -- see requirements.txt
# and .env.example for why those are deliberately not baked in here).
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

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embed_generate ./embed_generate
COPY model_train ./model_train

# Which Actor this image runs -- build with
#   docker build --build-arg ACTOR=embed_generate -t embed-generate .
#   docker build --build-arg ACTOR=model_train    -t model-train .
# A real deployment may instead split this into two separate Dockerfiles
# (Dockerfile.embed-generate / Dockerfile.model-train) once each Actor is
# actually registered with Tapis and their runtime needs diverge (e.g. only
# embed_generate needs the Clay checkpoint/GPU access) -- not done here since
# neither Actor has real logic yet to diverge over.
ARG ACTOR=embed_generate
ENV ACTOR=${ACTOR}

ENTRYPOINT ["sh", "-c", "python -m ${ACTOR}.main"]
