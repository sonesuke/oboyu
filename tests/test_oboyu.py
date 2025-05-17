import oboyu


def test_import():
    version = oboyu.__version__
    assert isinstance(version, str)
