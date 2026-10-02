"""
Artifact storage for the NES experiment suite.

Two responsibilities, both required by the safety rules in the handoff:

1.  Every expensive experiment writes a structured JSON artifact. Terminal
    output is never the only record (§25 rule 8, §22 step 4).
2.  Overwriting a completed artifact preserves the previous version
    (§25 rule 1).
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.experiments.paths import ARCHIVE_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Make torch/numpy scalars and non-finite floats JSON-serializable.

    ``Infinity``/``NaN`` are valid JSON for Python's writer but are not
    valid JSON for most consumers, and they silently turn a real
    measurement into an unreadable artifact. They are preserved as
    strings so the information is never lost (§25 rule 13).
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        return value

    # torch / numpy scalars expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    return str(value)


def archive_existing(path: Path, reason: str = "overwrite") -> Optional[Path]:
    """Move an existing artifact into ``results/_archive/`` before writing.

    Returns the archive path, or ``None`` when there was nothing to
    preserve. The archive keeps a timestamp so repeated overwrites of the
    same experiment do not collide.
    """
    if not path.exists():
        return None

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = ARCHIVE_DIR / f"{path.stem}__{stamp}__{reason}{path.suffix}"

    counter = 1
    while target.exists():
        target = (
            ARCHIVE_DIR
            / f"{path.stem}__{stamp}__{reason}_{counter}{path.suffix}"
        )
        counter += 1

    shutil.move(str(path), str(target))
    return target


def save_json(
    path: Path,
    data: Dict[str, Any],
    archive_previous: bool = True,
    reason: str = "overwrite",
) -> Path:
    """Write ``data`` to ``path``, archiving any previous version first."""
    path.parent.mkdir(parents=True, exist_ok=True)

    archived = None
    if archive_previous and path.exists():
        archived = archive_existing(path, reason=reason)

    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("generated_at", utc_now())

    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    if archived is not None:
        print(f"  [artifact] previous version archived -> {archived}")

    print(f"  [artifact] wrote {path}")
    return path


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON artifact, returning ``None`` when it is unusable.

    A corrupt or truncated artifact is treated as absent so the runner
    recomputes it rather than propagating garbage into the manifest.
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [artifact] unreadable, ignoring: {path} ({exc})")
        return None

    if not isinstance(data, dict):
        return None

    return data


def load_artifact_status(path: Path) -> Optional[str]:
    """Return the recorded ``status`` of an artifact, if it has one."""
    data = load_json(path)
    if data is None:
        return None
    return data.get("status")