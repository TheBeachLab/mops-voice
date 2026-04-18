"""Tests for image_attach: probabilistic attachment of file previews to tool results."""

import base64
import random
from pathlib import Path

import pytest
from PIL import Image

from mops_voice.image_attach import maybe_build_attachment


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    p = tmp_path / "logo.png"
    img = Image.new("RGB", (2000, 1500), color=(180, 60, 200))
    img.save(p, "PNG")
    return p


@pytest.fixture
def svg_path(tmp_path: Path) -> Path:
    p = tmp_path / "design.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>")
    return p


def test_returns_none_when_probability_is_zero(png_path: Path):
    rng = random.Random(0)
    assert maybe_build_attachment(str(png_path), probability=0.0, rng=rng) is None


def test_returns_block_when_probability_is_one(png_path: Path):
    rng = random.Random(0)
    block = maybe_build_attachment(str(png_path), probability=1.0, rng=rng)
    assert block is not None
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    # Confirm it's valid base64 of an actual image
    raw = base64.b64decode(block["source"]["data"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_is_downscaled_to_fit_max_dim(png_path: Path):
    rng = random.Random(0)
    block = maybe_build_attachment(
        str(png_path), probability=1.0, rng=rng, max_dim=512
    )
    assert block is not None
    raw = base64.b64decode(block["source"]["data"])
    from io import BytesIO
    out = Image.open(BytesIO(raw))
    assert max(out.size) <= 512
    # Aspect ratio preserved (within rounding): 2000:1500 = 4:3
    ratio = out.size[0] / out.size[1]
    assert abs(ratio - 4 / 3) < 0.05


def test_skips_non_png_files(svg_path: Path):
    rng = random.Random(0)
    assert maybe_build_attachment(str(svg_path), probability=1.0, rng=rng) is None


def test_returns_none_for_missing_file(tmp_path: Path):
    rng = random.Random(0)
    missing = tmp_path / "nope.png"
    assert maybe_build_attachment(str(missing), probability=1.0, rng=rng) is None


def test_probability_gate_uses_rng(png_path: Path):
    # With a seed that draws > 0.3 first, probability=0.3 should skip
    rng = random.Random(42)
    drawn = rng.random()
    assert drawn > 0.3  # confirm assumption about this seed
    rng = random.Random(42)
    assert maybe_build_attachment(str(png_path), probability=0.3, rng=rng) is None

    # With a seed that draws < 0.3 first, probability=0.3 should attach
    found_seed = None
    for s in range(100):
        if random.Random(s).random() < 0.3:
            found_seed = s
            break
    assert found_seed is not None
    rng = random.Random(found_seed)
    assert maybe_build_attachment(str(png_path), probability=0.3, rng=rng) is not None
