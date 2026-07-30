import os
import unittest
from unittest import mock

from tests import _stub_heavy_deps  # noqa: F401  (must run before embed_generate imports)

from embed_generate import main, webodm_client


class ResolveMaxWorkersTests(unittest.TestCase):
    """
    Decision 46: EMBED_MAX_WORKERS is explicit config (see main.py's own
    comment on DEFAULT_MAX_WORKERS for why this isn't auto-detected from
    os.cpu_count()/os.sched_getaffinity() -- unreliable on ls6's virtualized
    vm-small queue under Apptainer).
    """

    def test_defaults_to_serial_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('EMBED_MAX_WORKERS', None)
            self.assertEqual(main._resolve_max_workers(), 1)

    def test_reads_configured_value(self):
        with mock.patch.dict(os.environ, {'EMBED_MAX_WORKERS': '4'}):
            self.assertEqual(main._resolve_max_workers(), 4)

    def test_rejects_non_integer(self):
        with mock.patch.dict(os.environ, {'EMBED_MAX_WORKERS': 'lots'}):
            with self.assertRaises(RuntimeError):
                main._resolve_max_workers()

    def test_rejects_zero_or_negative(self):
        with mock.patch.dict(os.environ, {'EMBED_MAX_WORKERS': '0'}):
            with self.assertRaises(RuntimeError):
                main._resolve_max_workers()


class ProcessTileTests(unittest.TestCase):
    """
    _process_tile() is the unit of work run()'s ThreadPoolExecutor submits
    per tile -- these confirm it keeps the prior sequential loop's exact
    error-handling contract: a real coverage gap (TileNotFound) is a normal
    'skipped' outcome, but any other exception propagates (so
    future.result() in run() still surfaces it and fails the invocation).
    """

    VISIT = {
        'visit_id': 'v1', 'site_id': 's1', 'webodm_url': 'http://x',
        'project_pk': 1, 'webodm_task_id': 't1', 'webodm_jwt': 'jwt',
    }

    def test_tile_not_found_is_skipped_not_raised(self):
        with mock.patch.object(main, 'fetch_tile_pixels', side_effect=webodm_client.TileNotFound('nope')):
            status = main._process_tile(
                self.VISIT, zoom=19, x=1, y=1, encoder_id='e', capture_date=None,
                size='large', model_dir='/m', checkpoint_path='/c', session=None,
            )
        self.assertEqual(status, 'skipped')

    def test_real_error_propagates(self):
        with mock.patch.object(main, 'fetch_tile_pixels', side_effect=webodm_client.WebODMTilerError('boom')):
            with self.assertRaises(webodm_client.WebODMTilerError):
                main._process_tile(
                    self.VISIT, zoom=19, x=1, y=1, encoder_id='e', capture_date=None,
                    size='large', model_dir='/m', checkpoint_path='/c', session=None,
                )


if __name__ == '__main__':
    unittest.main()
