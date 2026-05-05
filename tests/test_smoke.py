def test_package_imports() -> None:
    import rba
    from rba.config import PROJ_ROOT, RANDOM_SEED

    assert PROJ_ROOT.exists()
    assert RANDOM_SEED == 12
    assert rba is not None
