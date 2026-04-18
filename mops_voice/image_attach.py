"""Probabilistic image attachments for tool results.

When `load_file` (or any tool that loads a PNG) returns, this module
optionally builds an Anthropic-API image content block from the file so
the LLM can see what was loaded and react to it (e.g., comment in
character before continuing the workflow).
"""

from __future__ import annotations

import base64
import io
import logging
import random
from pathlib import Path

from PIL import Image

log = logging.getLogger("mops_voice.image_attach")

_DEFAULT_RNG = random.Random()


def maybe_build_attachment(
    file_path: str,
    probability: float,
    *,
    max_dim: int = 512,
    rng: random.Random | None = None,
) -> dict | None:
    """Return an Anthropic image content block, or None to skip attachment.

    Skips silently (returns None) for: probability gate failure, missing
    file, non-PNG extension, or any read/encode error. The caller can
    treat None as "no image this time" without special-casing failures.
    """
    if probability <= 0:
        return None
    rng = rng or _DEFAULT_RNG
    if rng.random() >= probability:
        return None

    path = Path(file_path)
    if path.suffix.lower() != ".png" or not path.is_file():
        return None

    try:
        with Image.open(path) as img:
            img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        log.warning("could not prepare image attachment for %s", file_path, exc_info=True)
        return None

    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }
