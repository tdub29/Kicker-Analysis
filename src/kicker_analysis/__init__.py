"""Kicker analysis package.

`__init__` stays import-free on purpose. It used to pull in `pipeline`, which
imports `nfl_data_py` at module scope and raises if it is absent, so a missing
optional dependency made every module in the package unimportable, including the
ones that do not need it. Import what you need directly:

    from kicker_analysis.build_dataset import main as build
    from kicker_analysis.features import build_features
    from kicker_analysis.evaluate import run
"""
__all__ = ["build_dataset", "features", "fourth_down", "evaluate"]
