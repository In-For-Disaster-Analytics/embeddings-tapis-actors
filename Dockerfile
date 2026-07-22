# embeddings-tapis-actors
#
# Minimal image for the two Actors in this repo (embed_generate, model_train).
# Neither Actor has real logic yet -- see README "Status: skeleton only" --
# so this Dockerfile only needs to prove the image builds and can run either
# entrypoint; it is not tuned for the real Clay v1.5 inference workload yet
# (no GPU base image, no vendored claymodel/checkpoint -- see requirements.txt
# and .env.example for why those are deliberately not baked in here).
#
# NOTE on Tapis Actor image conventions: unlike this repo's sibling
# label-studio-tapis-auth (a Tapis POD, registered via tapis/register_pod.py,
# confirmed to work from a public GHCR image), Tapis ACTORS are a different
# Tapis subsystem (Abaco) with their own registration API. Whether Abaco
# accepts a GHCR image the same way Pods do has NOT been verified for this
# repo -- do not assume it does. No registration script exists in this repo
# yet (see README "Next steps" item 6); confirm the actual image-source
# requirement against Tapis's Actor docs before registering these Actors for
# real, rather than assuming the Pods precedent carries over unchanged.

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
