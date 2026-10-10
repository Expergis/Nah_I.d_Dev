from sensei.timing import RunStats, percentile


def test_percentile_bounds():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0
    assert percentile(vals, 50) == 3.0
    assert percentile([], 50) == 0.0


def test_summary_renders():
    stats = RunStats()
    stats.record_image("6215", "1751001268.jpg", 42.3)
    stats.record_area(
        camera="6215",
        image="1751001268.jpg",
        area_id="6215_1",
        wall_s=0.5,
        samples_ok=3,
        samples_total=3,
        cached_calls=1,
        prompt_tokens=1200,
        decode_tokens=12,
        prefill_ms=150.0,
        decode_ms=20.0,
    )
    text = stats.summary()
    assert "RUN TIMING SUMMARY" in text
    assert "Per image" in text
    assert "Per area" in text
    assert "Prefill/call" in text


def test_as_dict():
    stats = RunStats()
    stats.record_image("6215", "1751001268.jpg", 1.0)
    stats.record_area(
        camera="6215", image="1751001268.jpg", area_id="6215_1",
        wall_s=0.2, samples_ok=3, samples_total=3,
    )
    d = stats.as_dict()
    assert d["images"][0]["camera"] == "6215"
    assert d["areas"][0]["area_id"] == "6215_1"
    assert "total_s" in d
