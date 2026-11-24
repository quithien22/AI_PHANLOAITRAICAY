import io

import numpy as np
import pytest
from PIL import Image

from src.features import N_FEATURES, InvalidImageError, preprocess_image_bytes, to_rgb_on_white


def _png_bytes(size=(50, 80), color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_shape_and_finite():
    arr = preprocess_image_bytes(_png_bytes())
    assert arr.shape == (1, N_FEATURES)
    assert np.isfinite(arr).all()


def test_histogram_features_sum_to_one():
    arr = preprocess_image_bytes(_png_bytes())[0]
    # 3 histogram HSV (16 + 8 + 8 bin), mỗi histogram được chuẩn hoá để tổng = 1
    assert arr[:16].sum() == pytest.approx(1.0, abs=1e-5)
    assert arr[16:24].sum() == pytest.approx(1.0, abs=1e-5)
    assert arr[24:32].sum() == pytest.approx(1.0, abs=1e-5)


def test_different_colors_give_different_features():
    red = preprocess_image_bytes(_png_bytes(color=(255, 0, 0)))
    dark = preprocess_image_bytes(_png_bytes(color=(40, 25, 10)))
    assert not np.allclose(red, dark)


def test_preprocess_accepts_rgba():
    buf = io.BytesIO()
    Image.new("RGBA", (30, 30), (0, 255, 0, 128)).save(buf, format="PNG")
    assert preprocess_image_bytes(buf.getvalue()).shape == (1, N_FEATURES)


def test_transparent_pixels_become_white():
    # Điểm ảnh trong suốt có RGB ẩn là đen: phải thành trắng, không được thành đen
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    img.putpixel((5, 5), (255, 128, 0, 255))  # 1 điểm cam đục
    out = to_rgb_on_white(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((5, 5)) == (255, 128, 0)


def test_opaque_image_unchanged_by_white_compositing():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    assert to_rgb_on_white(img).getpixel((1, 1)) == (10, 20, 30)


def test_preprocess_rejects_non_image():
    with pytest.raises(InvalidImageError):
        preprocess_image_bytes(b"day khong phai anh")
