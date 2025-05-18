"""Basic tests for the oboyu package."""

import oboyu


def test_import() -> None:
    """Test that the package can be imported."""
    version = oboyu.__version__
    assert isinstance(version, str)

