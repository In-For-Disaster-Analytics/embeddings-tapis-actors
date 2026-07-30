import unittest
from unittest import mock

from tests import _stub_heavy_deps  # noqa: F401  (must run before embed_generate imports)

from embed_generate import clay

_MINIMAL_LINZ_YAML = (
    "linz:\n"
    "  gsd: 0.5\n"
    "  band_order: [red, green, blue]\n"
    "  bands:\n"
    "    wavelength: {red: 0.6, green: 0.55, blue: 0.45}\n"
    "    mean: {red: 1, green: 1, blue: 1}\n"
    "    std: {red: 1, green: 1, blue: 1}\n"
)


class LoadSensorMetadataCacheTests(unittest.TestCase):
    """
    embed_tile() is called once per tile (up to ~1764 per visit) from
    run()'s ThreadPoolExecutor -- these confirm the metadata.yaml read/parse
    only happens once per (model_dir, sensor), not on every tile.
    """

    def setUp(self):
        clay._sensor_metadata_cache.clear()

    def test_second_call_does_not_reopen_file(self):
        with mock.patch('builtins.open', mock.mock_open(read_data=_MINIMAL_LINZ_YAML)) as mocked_open:
            first = clay._load_sensor_metadata('/model', 'linz')
            second = clay._load_sensor_metadata('/model', 'linz')
        self.assertEqual(mocked_open.call_count, 1)
        self.assertIs(first, second)
        self.assertEqual(first['gsd'], 0.5)

    def test_different_model_dir_is_cached_separately(self):
        with mock.patch('builtins.open', mock.mock_open(read_data=_MINIMAL_LINZ_YAML)) as mocked_open:
            clay._load_sensor_metadata('/model-a', 'linz')
            clay._load_sensor_metadata('/model-b', 'linz')
        self.assertEqual(mocked_open.call_count, 2)


class DisableIntraopParallelismTests(unittest.TestCase):
    """
    Cross-tile parallelism comes from run()'s own ThreadPoolExecutor -- this
    must turn OFF PyTorch's intra-op threading so it doesn't oversubscribe
    on top of that (see disable_intraop_parallelism()'s own docstring).
    """

    def test_calls_set_num_threads_one(self):
        with mock.patch.object(clay, 'torch') as mocked_torch:
            clay.disable_intraop_parallelism()
        mocked_torch.set_num_threads.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()
