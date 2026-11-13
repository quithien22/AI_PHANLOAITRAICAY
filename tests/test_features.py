import io

import pytest
from PIL import Image

from src.features import InvalidImageError, preprocess_image_bytes


def _png_bytes(size=(50, 80), color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_shape_and_range():
    arr = preprocess_image_bytes(_png_bytes())
    assert arr.shape == (1, 224, 224, 3)
    assert arr.min() >= -1.0 and arr.max() <= 1.0


def test_preprocess_accepts_rgba():
    buf = io.BytesIO()
    Image.new("RGBA", (30, 30), (0, 255, 0, 128)).save(buf, format="PNG")
    assert preprocess_image_bytes(buf.getvalue()).shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    with pytest.raises(InvalidImageError):
        preprocess_image_bytes(b"day khong phai anh")
