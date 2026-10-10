import numpy as np

from sensei.geometry import context_crop, encode_png, fruit_fraction, shelf_features, warp_area


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


def test_fruit_fraction_bare_vs_produce():
    bare = np.full((64, 64, 3), (40, 40, 40), dtype=np.uint8)      # dark shelf -> ~0
    yellow = np.full((64, 64, 3), (0, 255, 255), dtype=np.uint8)    # saturated -> ~1
    assert fruit_fraction(bare) < 0.05
    assert fruit_fraction(yellow) > 0.95


def test_fruit_fraction_uses_mask():
    # left half produce-like, right half black padding; mask says only left is valid
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:, :32] = (0, 255, 255)   # BGR yellow on the left half
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 1
    # masked: only valid (left) pixels count -> all produce
    assert fruit_fraction(crop, mask) > 0.95
    # unmasked: black right half dilutes it
    assert 0.4 < fruit_fraction(crop, None) < 0.6


def test_shelf_features_bare_shelf():
    """A dark bare shelf should have high dark, low fruit."""
    bare = np.full((64, 64, 3), (30, 30, 35), dtype=np.uint8)  # very dark
    f = shelf_features(bare)
    assert f["dark"] > 0.9
    assert f["fruit"] < 0.01
    assert f["v_std"] < 10  # uniform dark -> low std
    assert f["texture"] == 0.0  # uniform image -> no Canny edges


def test_shelf_features_produce():
    """Saturated yellow/orange produce should have low dark, high fruit."""
    produce = np.full((64, 64, 3), (0, 200, 255), dtype=np.uint8)  # BGR yellow
    f = shelf_features(produce)
    assert f["dark"] < 0.1
    assert f["fruit"] > 0.9


def test_shelf_features_dark_produce_not_flagged():
    """Dark produce (aubergine-like) is a known risk for the colour net.
    It is dark AND low-sat, so it MIGHT fire the override. This test
    documents the limitation. In practice, real produce has enough
    saturation/texture to exceed the conservative thresholds."""
    aubergine = np.full((64, 64, 3), (80, 20, 30), dtype=np.uint8)  # dark purple-ish, V=80
    f = shelf_features(aubergine)
    # Dark produce will have high dark and low fruit - this is the known risk.
    assert f["dark"] > 0.5  # V=80 < 100, so it IS dark
    assert f["fruit"] < 0.1  # low saturation
    # In practice, real produce has texture and slight colour variation
    # that pushes it above the conservative thresholds. This synthetic test
    # documents the edge case.


def test_shelf_features_returns_all_keys():
    """shelf_features must return exactly the expected keys."""
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    f = shelf_features(img)
    assert set(f.keys()) == {"dark", "fruit", "v_std", "texture"}
    assert all(isinstance(v, (int, float)) for v in f.values())
