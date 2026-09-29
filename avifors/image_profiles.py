"""Administrator-controlled render profiles; public output dimensions stay stable."""

from __future__ import annotations

import io
import re

LEGACY_SIZES = ["512x512", "640x640", "512x768", "768x512"]


def dimensions(value):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{2,3}x[1-9][0-9]{2,3}", value):
        raise ValueError("invalid image size")
    width, height = map(int, value.split("x"))
    if any(x < 256 or x > 2048 or x % 32 for x in (width, height)) or width * height > 2097152:
        raise ValueError("image sizes must be multiples of 32, 256–2048 pixels and at most 2 megapixels")
    return width, height


def validate_profile(parameters):
    sizes = parameters.get("sizes", LEGACY_SIZES)
    if (
        not isinstance(sizes, list)
        or not sizes
        or not all(isinstance(size, str) for size in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("image sizes must be a nonempty unique list")
    for size in sizes:
        dimensions(size)
    if parameters.get("default_size", "640x640") not in sizes:
        raise ValueError("default image size must be allowed")
    renders = parameters.get("render_sizes", {})
    if not isinstance(renders, dict) or set(renders) - set(sizes):
        raise ValueError("render_sizes must map allowed output sizes")
    for output, render in renders.items():
        ow, oh = dimensions(output)
        rw, rh = dimensions(render)
        if ow * rh != oh * rw or rw < ow or rh < oh:
            raise ValueError("render size must preserve aspect ratio and be at least the output size")
    if parameters.get("negative_prompt_mode", "native") not in {"native", "instruction"}:
        raise ValueError("unknown negative prompt mode")


def resize_png(data, output_size, render_size):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as source:
        if source.format != "PNG" or source.size != tuple(render_size):
            raise ValueError("worker returned unexpected image dimensions")
        source.load()
        output = io.BytesIO()
        source.resize(tuple(output_size), Image.Resampling.LANCZOS).save(output, format="PNG")
        return output.getvalue()
