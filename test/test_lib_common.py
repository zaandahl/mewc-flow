import unittest
from unittest.mock import patch, mock_open
import os
from lib_common import read_yaml, update_config_from_env, model_img_size_mapping, setup_strategy, NullStrategy

class TestLibCommon(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="key: value")
    def test_read_yaml(self, mock_file):
        result = read_yaml('dummy_path.yaml')
        self.assertEqual(result, {'key': 'value'})
        mock_file.assert_called_with('dummy_path.yaml', 'r')

    @patch.dict(os.environ, {'TEST_KEY': '123', 'TEST_LIST': '1,2,3'}, clear=True)
    def test_update_config_from_env(self):
        config = {'TEST_KEY': 0, 'TEST_LIST': [0, 0, 0]}
        updated_config = update_config_from_env(config)
        self.assertEqual(updated_config['TEST_KEY'], 123)
        self.assertEqual(updated_config['TEST_LIST'], [1, 2, 3])

    def test_model_img_size_mapping(self):
        self.assertEqual(model_img_size_mapping('ENB0'), 224)
        self.assertEqual(model_img_size_mapping('ENB2'), 260)
        self.assertEqual(model_img_size_mapping('ENS'), 384)
        self.assertEqual(model_img_size_mapping('ENM'), 480)
        self.assertEqual(model_img_size_mapping('ENXL'), 512)
        self.assertEqual(model_img_size_mapping('CNP'), 288)
        self.assertEqual(model_img_size_mapping('CNT'), 384)
        self.assertEqual(model_img_size_mapping('ViTT'), 384)
        self.assertEqual(model_img_size_mapping('UnknownModel'), 384)

    @patch('lib_common.devices', return_value=['cpu'])
    @patch('lib_common.distribution')
    def test_setup_strategy_cpu(self, mock_distribution, mock_devices):
        # Test setup_strategy when only CPU is available
        strategy = setup_strategy()
        self.assertIsInstance(strategy, NullStrategy)
        self.assertFalse(mock_distribution.DataParallel.called)

    @patch('lib_common.devices', return_value=['cuda:0', 'cuda:1'])
    @patch('lib_common.distribution')
    def test_setup_strategy_gpu(self, mock_distribution, mock_devices):
        # Test setup_strategy when GPU is available
        strategy = setup_strategy()
        mock_distribution.DataParallel.assert_called_with(devices=['cuda:0', 'cuda:1'])
        self.assertTrue(mock_distribution.DataParallel.called)

if __name__ == '__main__':
    unittest.main()
