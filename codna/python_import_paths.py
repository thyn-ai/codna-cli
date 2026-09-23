from __future__ import annotations

import re
import sys
from pathlib import Path


def site_packages_python_version(path: Path) -> tuple[int, int] | None:
    for parent in (path, *path.parents):
        match = re.fullmatch(r"python(\d+)\.(\d+)", parent.name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def compatible_engine_site_packages(engine_dir: Path) -> list[Path]:
    current_python = sys.version_info[:2]
    return sorted(
        (
            site_packages
            for site_packages in (engine_dir / ".venv" / "lib").glob("python*/site-packages")
            if site_packages_python_version(site_packages) in {None, current_python}
        ),
        reverse=True,
    )
