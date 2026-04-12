"""
Load task packages under ``tasks/<dirname>/`` when the folder name is not a valid
Python module identifier (e.g. ``1_day_move``, ``30_day_move``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_task_subpackage(dirname: str, qualname: str) -> ModuleType:
    """
    Load ``tasks/<dirname>/__init__.py`` as ``qualname``, with sibling modules
    (``spec.py``, ``grader.py``) importable as submodules.
    """
    root = Path(__file__).resolve().parent / dirname
    init_path = root / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"Missing task package: {init_path}")

    spec = importlib.util.spec_from_file_location(
        qualname,
        init_path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task package from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module
