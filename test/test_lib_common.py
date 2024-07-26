import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import yaml
import logging  # Import logging for the test
from lib_common import configure_logging, read_yaml, update_config_from_env, model_img_size_mapping, setup_strategy

class TestLibCommon(unittest.TestCase):

    @patch('lib_common.logging.getLogger')
    @patch('lib_common.warnings')
    @patch.dict(os.environ, {}, clear=True)
    def test_configure_logging(self, mock_warnings, mock_getLogger):
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        
        configure_logging()
        
        mock_getLogger.assert_called_with('tensorflow')
        mock_logger.setLevel.assert_called_with(logging.ERROR)
        mock_warnings.simplefilter.assert_any_call(action='ignore', category=FutureWarning)
        mock_warnings.simplefilter.assert_any_call(action='ignore', category=Warning)
        mock_warnings.filterwarnings.assert_called_with("ignore")
        self.assertEqual(os.environ['TF_CPP_MIN_LOG_LEVEL'], '3')

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
        # Mock the NullStrategy within the setup_strategy function
        with patch('lib_common.setup_strategy.__globals__["NullStrategy"]', new_callable=MagicMock) as MockNullStrategy:
            strategy = setup_strategy()
            self.assertIsInstance(strategy, MockNullStrategy)
            self.assertFalse(mock_distribution.DataParallel.called)

    @patch('lib_common.devices', return_value=['cuda:0', 'cuda:1'])
    @patch('lib_common.distribution')
    def test_setup_strategy_gpu(self, mock_distribution, mock_devices):
        strategy = setup_strategy()
        mock_distribution.DataParallel.assert_called_with(devices=['cuda:0', 'cuda:1'])
        self.assertTrue(mock_distribution.DataParallel.called)

if __name__ == '__main__':
    unittest.main()
