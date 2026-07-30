"""
Stubs torch/psycopg2/claymodel in sys.modules before embed_generate's own
modules are imported.

Neither is installed outside the Docker image (Dockerfile.embed-generate
bakes in torch + the vendored claymodel package; psycopg2-binary comes from
requirements.txt but isn't part of a plain local dev environment) -- but
embed_generate.clay/db import them at module load time regardless of
whether a given test actually exercises real model/DB behavior. Import this
module first, before importing anything under embed_generate, in any test
that needs main.py/clay.py/db.py to import successfully.
"""

import sys
from unittest import mock

for _name in (
    'torch',
    'psycopg2',
    'claymodel',
    'claymodel.finetune',
    'claymodel.finetune.embedder',
    'claymodel.finetune.embedder.factory',
):
    sys.modules.setdefault(_name, mock.MagicMock())
