"""Repository paths shared by current commands and historical run readers."""

from pathlib import Path


def config_path(value):
    """Resolve pre-refactor config references without rewriting archived metadata."""
    path = Path(value)
    if path.exists():
        return path
    if len(path.parts) == 2 and path.parts[0] == "configs":
        for stage in ("p1", "p2"):
            if path.name.startswith(stage + "-"):
                return Path("configs") / stage / path.name
    return path
