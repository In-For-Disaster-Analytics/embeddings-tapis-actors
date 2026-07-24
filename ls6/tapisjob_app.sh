#!/bin/bash

# embed-generate execution script for Tapis Jobs on TACC Lonestar6 (ls6).
#
# Design spec Decision 45: embed-generate's Actor (Abaco) cannot provision a
# worker for this workload's image size (confirmed: both an ~11.4GB image
# with the Clay v1.5 checkpoint baked in, and a ~6.21GB image without it,
# failed identically -- zero workers ever provisioned). This script runs
# the SAME GHCR image as a real Tapis Job on ls6 instead, mirroring
# nodeodm-ls6/tapisjob_app.sh's proven "pull a large Docker image straight
# onto the HPC node via Apptainer" pattern -- an HPC compute node's
# disk/memory envelope is not Abaco's.
#
# Much shorter than nodeodm-ls6's own script: no TAP/reverse-tunnel web
# access (that's for NodeODM's long-running interactive service; this is a
# one-shot batch job that exits when embed_generate.main finishes), no
# multi-node/cluster coordination, no GPU detection (Decision 45: starts
# CPU-only on the vm-small queue by user decision -- add GPU detection back,
# mirroring nodeodm-ls6's nvidia-smi/$SLURM_JOB_PARTITION check, if/when a
# GPU queue is used).
#
# Real Abaco message contract, UNCHANGED: embed_generate/main.py's
# read_actor_message() reads its entire invocation payload from a single
# MSG environment variable (a JSON string) -- that was Abaco's convention,
# but nothing about main.py itself is Abaco-specific once MSG is set. This
# script does not construct MSG itself -- Tapis's own parameterSet.
# envVariables (set at job-submission time by embeddings_client.py's
# apply_embed_generate(), see Decision 45) already puts it in this script's
# environment before it runs, identically to how Abaco did.

set -e

if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

EMBED_GENERATE_IMAGE=${EMBED_GENERATE_IMAGE:-ghcr.io/in-for-disaster-analytics/embeddings-tapis-actors:embed-generate-latest}

module load tacc-apptainer

# Cache the pulled SIF within this job's own working directory -- mirrors
# nodeodm-ls6/tapisjob_app.sh's own WORK_DIR-scoped caching exactly (not a
# cross-job persistent cache; each Tapis Job gets a fresh
# execSystemExecDir, so this only avoids re-pulling if apptainer itself
# retries within one job run). A persistent cross-job cache (e.g. under
# $WORK) is a real, later optimization -- not implemented here, matching
# the precedent's actual behavior rather than inventing an unproven scheme.
SIF_PATH="$(pwd)/embed-generate.sif"
echo "Ensuring local SIF image at: $SIF_PATH"
if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling embed-generate image into local SIF: docker://$EMBED_GENERATE_IMAGE"
    apptainer pull "$SIF_PATH" "docker://$EMBED_GENERATE_IMAGE" || {
        echo "ERROR: Failed to pull embed-generate image to $SIF_PATH"
        exit 1
    }
else
    echo "Using existing SIF image at $SIF_PATH"
fi

echo "SIF image details:"
ls -lh "$SIF_PATH" || true

if [ -z "${MSG:-}" ]; then
    echo "ERROR: MSG environment variable is not set -- Tapis's parameterSet.envVariables should have set it at job submission time (see apply_embed_generate() in coreplugins/embeddings/embeddings_client.py, WebODM repo)."
    exit 1
fi

echo "Running embed-generate..."
set +e
apptainer exec \
    --env MSG="$MSG" \
    --env EMBEDDINGSDB_URL="$EMBEDDINGSDB_URL" \
    --env WEBODM_URL="${WEBODM_URL:-https://webodm.tacc.utexas.edu}" \
    "$SIF_PATH" \
    python -m embed_generate.main
EXIT_CODE=$?

echo "embed-generate exited with code $EXIT_CODE"
exit $EXIT_CODE
