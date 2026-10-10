import numpy as np

from sensei.geometry import context_crop, encode_png, warp_area


def _image() -> np.ndarray:
    # 200x200 gradient image so warped content is checkable
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 200).astype(np.uint8)[None, :]
    return img


def test_warp_full_image_quad():
    corners = ((0, 0), (199, 0), (199, 199), (0, 199))
    crop, mask, visible = warp_area(_image(), corners, 64, 64)
    assert crop.shape == (64, 64, 3)
    assert visible > 0.99
    assert mask.min() == 1  # everything valid


def test_warp_partial_off_image_quad():
    # One corner off-image (negative) -> part of the crop is padding
    corners = ((-50, 0), (100, 0), (100, 100), (-50, 100))
    crop, mask, visible = warp_area(_image(), corners, 64, 64)
    assert 0.0 < visible < 1.0
    # padding pixels are black
    assert (crop[mask == 0] == 0).all()


def test_context_crop_clipped_to_image():
    corners = ((10, 10), (60, 10), (60, 60), (10, 60))
    ctx = context_crop(_image(), corners, padding=0.5)
    assert ctx is not None
    # padded box must be strictly larger than the area and inside the image
    assert ctx.shape[0] > 50 and ctx.shape[1] > 50
    assert ctx.shape[0] <= 200 and ctx.shape[1] <= 200


def test_encode_png_roundtrip():
    import cv2

    img = _image()
    data = encode_png(img)
    # encode_png() now emits JPEG (smaller payloads for the VLM)
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == img.shape
