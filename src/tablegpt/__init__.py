from __future__ import annotations

import site
import warnings
from pathlib import Path


def _find_tablegpt_ipykernel_profile_dir(path):
    possible_installation_location = Path(path).parents[2]
    possible_profile_dir = Path(possible_installation_location, "share", "ipykernel", "profile", "tablegpt")

    _startup_folder = Path(possible_profile_dir, "startup")
    try:
        if next(_startup_folder.glob(r"*-udfs.py")):
            return str(possible_profile_dir)
    except StopIteration:
        return


try:
    DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR: str | None = next(
        path
        for path in map(_find_tablegpt_ipykernel_profile_dir, [*site.getsitepackages(), site.getusersitepackages()])
        if path is not None
    )

except StopIteration:
    # Means not found.
    msg = """Unable to find tablegpt ipykernel. If you need to use a local kernel,
please use `pip install -U tablegpt-agent[local]` to install the necessary dependencies.
For more issues, please submit an issue to us https://github.com/tablegpt/tablegpt-agent/issues."""
    warnings.warn(msg, stacklevel=2)
    DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR = None
