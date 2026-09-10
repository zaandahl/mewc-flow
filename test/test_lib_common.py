import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from lib_common import (get_mod, model_img_size_mapping, NullStrategy, read_yaml,
                        setup_strategy, update_config_from_env)


def test_read_yaml_rejects_duplicate_keys(tmp_path):
    path=tmp_path/'classes.yaml'
    path.write_text('0: quoll\n0: devil\n')
    with pytest.raises(ValueError,match='Duplicate'): read_yaml(path)


@pytest.mark.parametrize('value,expected',[('True',True),('False',False),('true',True),('false',False),('1',True),('0',False),('yes',True),('off',False)])
def test_boolean_parsed_before_integer(value,expected):
    assert update_config_from_env({'FLAG':False},{'FLAG':value})['FLAG'] is expected


@pytest.mark.parametrize('value',['truthy','2','none',''])
def test_invalid_boolean_rejected(value):
    with pytest.raises(ValueError): update_config_from_env({'FLAG':False},{'FLAG':value})


def test_integer_list_and_string_types():
    assert update_config_from_env({'N':1,'L':[1],'S':'x'},{'N':'12','L':'1,2,3','S':'001'}) == {'N':12,'L':[1,2,3],'S':'001'}
    with pytest.raises(ValueError): update_config_from_env({'N':1},{'N':'1.5'})
    with pytest.raises(ValueError,match='Unknown'): update_config_from_env({'N':1},{'DRAW':'true'})


def test_update_config_from_actual_environment_ignores_unrelated_variables(monkeypatch):
    monkeypatch.delenv('MEWC_TEST_CONFIG_VALUE', raising=False)
    monkeypatch.setenv('UNRELATED_MEWC_TEST_OPTION', 'not-a-config-value')
    assert update_config_from_env({'MEWC_TEST_CONFIG_VALUE': 1}) == {'MEWC_TEST_CONFIG_VALUE': 1}


@pytest.mark.parametrize('filename,expected', [
    ('ENB0_classifier.keras', 'ENB0'),
    ('ENB2_classifier.keras', 'ENB2'),
    ('ENXL_classifier.keras', 'ENXL'),
    ('ViTT_classifier.keras', 'ViTT'),
    ('ViTS_classifier.keras', 'ViTS'),
    ('ViTB_classifier.keras', 'ViTB'),
    ('ViTL_classifier.keras', 'ViTL'),
    ('EN0_classifier.keras', 'EN0'),
])
def test_get_mod_preserves_supported_keras_alias_prefixes(filename, expected):
    assert get_mod(f'/models/{filename}') == expected


def test_get_mod_keeps_non_keras_filename_parsing():
    assert get_mod('/models/ENB0_classifier.h5') == 'ENB0_classifier'


@pytest.mark.parametrize('name,size',[('ENB0',224),('EN0',224),('ENB2',260),('EN2',260),('ENS',384),('ENM',480),('ENL',480),('ENXL',512),('ENX',512),('CNP',288),('CNT',384),('ViTT',384),('VTL',384)])
def test_exact_architecture_aliases(name,size):
    assert model_img_size_mapping(name)==size


@pytest.mark.parametrize('name',['UnknownModel','VTLtypo','ENS_other','ENB'])
def test_unknown_architecture_rejected(name):
    with pytest.raises(ValueError): model_img_size_mapping(name)


@pytest.mark.parametrize('devices',[['cpu'],['cuda:0','cuda:1']])
def test_strategy_lazy_import(monkeypatch,devices):
    distribution=SimpleNamespace(DataParallel=Mock())
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(devices=lambda:devices))
    monkeypatch.setitem(sys.modules,'keras',SimpleNamespace(distribution=distribution))
    strategy=setup_strategy()
    if devices==['cpu']: assert isinstance(strategy,NullStrategy)
    else: distribution.DataParallel.assert_called_once_with(devices=devices)
