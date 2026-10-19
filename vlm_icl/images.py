"""Image input resolution: PIL images, file paths, bytes and Arrow image structs."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Union

from PIL import Image


def resolve_image(x: Any, prefer_pil: bool = False) -> Union[Image.Image, str]:
    """Convert any supported image input to a PIL Image or file path string."""
    def _is_url_like(s: str) -> bool:
        return s.startswith(("http://", "https://", "file://", "data:"))

    def _open_local_path(p: str) -> Image.Image:
        return Image.open(p).convert("RGB")

    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str):
        if prefer_pil and not _is_url_like(x) and Path(x).is_file():
            return _open_local_path(x)
        return x
    if isinstance(x, bytes):
        return Image.open(BytesIO(x)).convert("RGB")
    if isinstance(x, dict):
        b = x.get("bytes", None)
        if b is not None:
            if isinstance(b, memoryview):
                b = b.tobytes()
            elif isinstance(b, bytearray):
                b = bytes(b)
            try:
                return Image.open(BytesIO(b)).convert("RGB")
            except Exception:
                # Fall back to path if bytes are missing/corrupt
                p = x.get("path")
                if p and prefer_pil and not _is_url_like(p) and Path(p).is_file():
                    return _open_local_path(p)
                if p:
                    return p
                raise
        p = x.get("path")
        if p:
            if prefer_pil and not _is_url_like(p) and Path(p).is_file():
                return _open_local_path(p)
            return p
        raise ValueError("Unsupported image dict: expected 'bytes' or 'path'")

    try:
        import pyarrow as pa
    except ImportError:
        pa = None

    if pa is not None and isinstance(x, pa.StructScalar):
        try:
            b_scalar = x["bytes"]
        except Exception:
            b_scalar = None
        b = b_scalar.as_py() if getattr(b_scalar, "is_valid", False) else None
        if b is not None:
            try:
                return Image.open(BytesIO(b)).convert("RGB")
            except Exception:
                # Fall back to path if bytes are missing/corrupt
                try:
                    p_scalar = x["path"]
                except Exception:
                    p_scalar = None
                p = p_scalar.as_py() if getattr(p_scalar, "is_valid", False) else None
                if p and prefer_pil and not _is_url_like(p) and Path(p).is_file():
                    return _open_local_path(p)
                if p:
                    return p
                raise

        try:
            p_scalar = x["path"]
        except Exception:
            p_scalar = None
        p = p_scalar.as_py() if getattr(p_scalar, "is_valid", False) else None
        if p:
            if prefer_pil and not _is_url_like(p) and Path(p).is_file():
                return _open_local_path(p)
            return p
        raise ValueError("Unsupported pyarrow image struct: missing bytes/path")

    raise TypeError(f"Unsupported image type: {type(x)}")


def to_pil(image_input: Any) -> Image.Image:
    """Normalize any supported image input to a PIL Image."""
    resolved = resolve_image(image_input, prefer_pil=True)
    if isinstance(resolved, Image.Image):
        return resolved
    # resolve_image returns a path string when it cannot open the input as PIL
    return Image.open(resolved).convert("RGB")


def pil_to_data_url(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 JPEG data URL."""
    buffer = BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
