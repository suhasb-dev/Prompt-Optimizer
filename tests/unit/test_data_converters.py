import pytest
from gepa_optimizer.data.converters import UniversalConverter
from gepa_optimizer.utils.exceptions import DatasetError

def test_converter_file_not_found():
    converter = UniversalConverter()
    with pytest.raises(DatasetError):
        converter.convert('non_existent_file.csv')

def test_converter_unsupported_file_type():
    converter = UniversalConverter()
    with pytest.raises(DatasetError):
        converter.convert('test.unsupported')
