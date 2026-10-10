from sensei.output import make_skeleton, save_output, validate_output
from sensei.planogram import Planogram


SAMPLE = {
    "Frutas e Legumes": {
        "cameras": {
            "6215": {
                "products": [
                    {
                        "area_id": "6215_2",
                        "product_id": "11908",
                        "product_name": "BANANA CAT I CNT KG",
                        "image_coordinates": {
                            "top_left": [299, 350],
                            "top_right": [464, 347],
                            "bottom_right": [479, 438],
                            "bottom_left": [331, 438],
                        },
                    }
                ]
            }
        }
    }
}


def _planogram(tmp_path) -> Planogram:
    path = tmp_path / "planogram.json"
    import json

    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return Planogram.load(path)


def test_planogram_loads_sample(tmp_path):
    pg = _planogram(tmp_path)
    assert pg.cameras() == ["6215"]
    assert pg.area_ids("6215") == ["6215_2"]
    area = pg.areas_for("6215")[0]
    assert area.product_name == "BANANA CAT I CNT KG"
    assert area.corners == ((299, 350), (464, 347), (479, 438), (331, 438))


def test_skeleton_and_validation(tmp_path):
    pg = _planogram(tmp_path)
    images = {"6215": ["1751052241.jpg", "1751052301.jpg"]}
    out = make_skeleton(pg, images)
    assert set(out["6215"]) == set(images["6215"])
    for img in images["6215"]:
        assert set(out["6215"][img]) == {"6215_2"}
        entry = out["6215"][img]["6215_2"]
        assert entry["stock_level"] == "empty"
        assert entry["stock_value"] == 0.0
        assert entry["planogram_compliant"] is True
    assert validate_output(out, pg, images) == []


def test_validation_catches_missing_area(tmp_path):
    pg = _planogram(tmp_path)
    images = {"6215": ["1751052241.jpg"]}
    out = make_skeleton(pg, images)
    del out["6215"]["1751052241.jpg"]["6215_2"]
    problems = validate_output(out, pg, images)
    assert any("missing area" in p for p in problems)


def test_validation_catches_bad_values(tmp_path):
    pg = _planogram(tmp_path)
    images = {"6215": ["1751052241.jpg"]}
    out = make_skeleton(pg, images)
    out["6215"]["1751052241.jpg"]["6215_2"] = {
        "planogram_compliant": "yes",
        "stock_level": "full",
        "stock_value": 0.0,  # mismatched with "full"
    }
    problems = validate_output(out, pg, images)
    assert any("stock_value mismatch" in p for p in problems)
    assert any("bad planogram_compliant" in p for p in problems)


def test_save_output(tmp_path):
    import json

    path = tmp_path / "out" / "predictions.json"
    save_output({"6215": {}}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"6215": {}}
