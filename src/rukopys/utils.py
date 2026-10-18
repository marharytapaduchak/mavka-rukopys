"""JSON, hashing, and bounding-box utilities."""

import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    """Read a UTF-8 JSON file and return its contents."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    """Write JSON through a temporary file and atomically replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def digest(value):
    """Compute a SHA-256 hash of a deterministically serialized JSON value."""
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def file_digest(path):
    """Compute a file's SHA-256 hash by reading it in chunks."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def valid_bbox(bbox, width, height):
    """Validate bounding-box coordinates and return them as floats."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must have four coordinates")

    if any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in bbox
    ):
        raise ValueError("Invalid bbox values")

    x1, y1, x2, y2 = bbox

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox outside image: {bbox}")

    return [float(x1), float(y1), float(x2), float(y2)]
