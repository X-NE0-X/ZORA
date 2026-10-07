"""All regression cases execute the sole current mathematical contract."""
import pytest

pytest_plugins = ("tests.regression_support",)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    from tests.regression_support import partition_contracts
    partition_contracts(items)

